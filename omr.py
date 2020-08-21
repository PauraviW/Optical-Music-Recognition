#!/usr/local/bin/python3

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from scipy.signal import convolve2d
import os
import sys
from scipy import fftpack

def padding(image, kernel):
    x, y = kernel.shape
    bottom = int(math.pow(2, math.ceil(math.log2(image.shape[0]))) - image.shape[0])
    right = int(math.pow(2, math.ceil(math.log2(image.shape[1]))) - image.shape[1])
    image = np.pad(image, ((x,bottom - x), (y,right - y)), mode='constant')
    
    bottom = image.shape[0] - kernel.shape[0]
    right = image.shape[1] - kernel.shape[1]
    kernel = np.pad(kernel, ((0,bottom), (0,right)), mode='constant')
    
    return image, kernel


def convolve2D_fourier(im, kernel):
    image, kernel = padding(im, kernel)
    
    im_ft = fftpack.fft2(image)
    kernel_ft = fftpack.fft2(kernel)
    
    im_ft_kernel_ft = im_ft * kernel_ft
    
    convolution = fftpack.ifft2(im_ft_kernel_ft)
    
    convolution = (np.log(abs(convolution))* 255 / 
             np.amax(np.log(abs(convolution)))).astype(np.uint8)

    return convolution[: im.shape[0], : im.shape[1]]


def template_matching(image, template):
    image[image > 1] = 1
    template[template > 1] = 1
    D = np.full(image.shape, np.inf)
    y = np.argwhere(image == 1)
    for i in range(D.shape[0]):
        pbar.update(1)
        x = np.array([[i, j] for j in range(D.shape[1])])
        D[i] = calculate_euclidean_distance(x, y).ravel()
    return convolve2D_fourier(D, template)


def calculate_euclidean_distance(x, y):  # one vs all
    distances = -2 * np.matmul(x, y.T)
    distances += np.sum(x ** 2, axis=1)[:, np.newaxis]
    distances += np.sum(y ** 2, axis=1)
    distances[distances == 0] = np.max(distances)
    return np.amin(distances, keepdims=True, axis = 1)

def get_patch(data_matrix, row_center, col_center, patch_len_row, patch_len_col, padding_value):
    row_midpoint = int((patch_len_row) / 2)
    col_midpoint = int((patch_len_col) / 2)

    large_matrix_padded = np.pad(data_matrix, ((row_midpoint,), (col_midpoint,)), 'constant', constant_values=padding_value)

    # the origin has shifted from 0,0 to -row_midpoint, -col_midpoint
    row_center_shifted = row_center + row_midpoint
    col_center_shifted = col_center + col_midpoint

    row_start = row_center_shifted - row_midpoint
    col_start = col_center_shifted - col_midpoint

    row_end = row_start + patch_len_row
    col_end = col_start + patch_len_col
    return large_matrix_padded[row_start:row_end, col_start:col_end]


def cross_correlation(image_data, kernel):
    height = image_data.shape[0]
    width = image_data.shape[1]
    n_rows_patch = kernel.shape[0]
    n_cols_patch = kernel.shape[1]
    new_image = np.array([[np.sum(np.asarray(get_patch(image_data, row, col, n_rows_patch, n_cols_patch, 1)) * kernel)
                           for col in range(width)] for row in range(height)])
    return new_image


def convolution(image_data, kernel, optimize=False):
    if optimize:
        return convolve2d(image_data, kernel)
    return cross_correlation(image_data, np.flip(kernel))


def create_edge_map(image_data):
    h_gradient = horizontal_edge_map(image_data)
    v_gradient = vertical_edge_map(image_data)
    return v_gradient, min_max_normalization(to_bw(np.sqrt(np.square(h_gradient) + np.square(v_gradient))))


def horizontal_edge_map(image_data):
    horizontal_filter_1 = np.array([1, 2, 1]).reshape(-1, 1)
    horizontal_filter_2 = np.array([1, 0, -1]).reshape(-1, 1)
    return convolution(convolution(image_data, horizontal_filter_1.T), horizontal_filter_2)


def vertical_edge_map(image_data):
    vertical_filter_1 = np.array([1, 0, -1]).reshape(-1, 1)
    vertical_filter_2 = np.array([1, 2, 1]).reshape(-1, 1)
    return convolution(convolution(image_data, vertical_filter_1.T), vertical_filter_2)


def min_max_normalization(data):
    return (data - data.min()) / (data.max() - data.min())


def hough_transform(edge_map):
    n_rows = edge_map.shape[0]
    hough_accumulator = np.zeros((n_rows, 50))
    for col in range(edge_map.shape[1]):
        starting_candidates = {row: True for row in range(edge_map.shape[0])}
        for row in range(1, edge_map.shape[0]):
            # New starting row candidate
            if starting_candidates[row] and edge_map[row, col] == 255:
                starting_candidates[row] = False
                prev_white_pixel_row = row
                spacing_candidate = 0
                for next_row in range(row, min(row + 50, n_rows)):
                    if edge_map[next_row, col] == 255:
                        spacing = next_row - prev_white_pixel_row
                        if spacing < 2:
                            prev_white_pixel_row = next_row
                            starting_candidates[next_row] = False
                        else:
                            spacing_candidate = next_row - prev_white_pixel_row
                        hough_accumulator[row, spacing_candidate] += 1
    row_with_vote = {row: np.max(hough_accumulator, axis=1)[row] for row in range(hough_accumulator.shape[0])}
    staff_lines = sorted(map(lambda x: x[0], sorted(row_with_vote.items(), key=lambda x: x[1], reverse=True)[:10]))
    return staff_lines


def image_as_data(path, height=None):
    im = Image.open(path).convert('L')
    if not height:
        return np.asarray(im).astype("float")
    conversion_ratio = height / im.height
    new_width = int(conversion_ratio * im.width)
    resized_image = im.resize(size=(new_width, height))
    im.close()
    return np.asarray(resized_image).astype("float")


def template_matching_using_hamming(image_data, template_data):
    part_1 = cross_correlation(image_data, template_data)
    part_2 = cross_correlation(1 - image_data, 1 - template_data)
    return 255 * min_max_normalization(part_1 + part_2)


def to_bw(image_data, threshold = 200):
    bw_image = np.array(image_data)
    bw_image[bw_image > threshold] = 255
    bw_image[bw_image != 255] = 0
    return bw_image

def get_matches(matched_image, threshold):
    threshold_image = to_bw(matched_image, threshold)
    rows, columns = np.where(threshold_image == 255)
    return rows, columns


def remove_padding(matrix):
    matrix_without_padding = np.array(matrix)
    matrix_without_padding[0:10] = 0
    matrix_without_padding[::-1][0:10] = 0
    matrix_without_padding.T[0:10] = 0
    matrix_without_padding.T[::-1][0:10] = 0
    return matrix_without_padding


if __name__ == "__main__":
    image_path = sys.argv[1]
    file_name = image_path.split("/")[-1].split(".")[0]
    image_data = image_as_data(image_path)

    if os.path.exists(f"test-images/{file_name}_h_edges.png"):
        image_h_edge_map = min_max_normalization(image_as_data(f"test-images/{file_name}_h_edges.png"))
    else:
        image_h_edge_map = min_max_normalization(to_bw(horizontal_edge_map(image_data)))
        Image.fromarray(image_h_edge_map * 255).convert("L").save(f"test-images/{file_name}_h_edges.png")

    if os.path.exists(f"test-images/{file_name}_staves.txt"):
       staves = np.loadtxt(f"test-images/{file_name}_staves.txt").astype('uint16')
    else:
        staves = hough_transform(image_h_edge_map * 255)
        np.savetxt(f"test-images/{file_name}_staves.txt", staves)


    staff_spacing = int(np.median(np.concatenate([np.diff(staves[start:start + 4]) for start in range(0, len(staves), 5)])))
    trebel_staves = np.concatenate([np.arange(staves[start], staves[start] + staff_spacing * 4, staff_spacing) for start in range(0, len(staves), 10)])
    bass_staves = np.concatenate([np.arange(staves[start], staves[start] + staff_spacing * 4, staff_spacing)  for start in range(5, len(staves), 10)])

    head_template_path = "test-images/template1.png"
    head_template_data = image_as_data(head_template_path, staff_spacing)
    head_matches = remove_padding(template_matching_using_hamming(min_max_normalization(to_bw(image_data)),
                                                         min_max_normalization(to_bw(head_template_data))))

    head_match_row, head_match_column = np.where(head_matches == 255)

    image_v_edge_map = min_max_normalization(to_bw(vertical_edge_map(image_data)))

    rest_1_template_path = "test-images/template2.png"
    rest_1_template_data = image_as_data(rest_1_template_path)
    rest_1_v_edge_map = min_max_normalization(to_bw(vertical_edge_map(rest_1_template_data)))

    rest_1_matches = remove_padding(to_bw(template_matching_using_hamming(min_max_normalization(to_bw(image_data)), min_max_normalization(to_bw(rest_1_template_data))), 210))
    rest_1_row, rest_1_column = np.where(rest_1_matches == 255)

    rest_2_template_path = "test-images/template3.png"
    rest_2_template_data = image_as_data(rest_2_template_path)
    rest_2_v_edge_map = min_max_normalization(to_bw(vertical_edge_map(rest_2_template_data)))

    rest_2_matches = remove_padding(to_bw(template_matching_using_hamming(min_max_normalization(to_bw(image_data)), min_max_normalization(to_bw(rest_2_template_data))), 210))
    rest_2_row, rest_2_column = np.where(rest_2_matches == 255)

    detected_image = Image.open(image_path)
    scratchpad = ImageDraw.Draw(detected_image)
    font = ImageFont.truetype("/usr/share/fonts/open-sans/OpenSans-Regular.ttf", 16)

    treble_notes_on_line = ["F", "D", "B", "G", "E"]
    treble_notes_between_lines = ["G", "E", "C", "A", "F", "D"]


    detections = []

    # Head matches
    thresholds = np.arange(255, 225, -5)
    confidences = min_max_normalization(np.arange(thresholds.size))[::-1] * 100
    already_matched_pixels = set()
    for idx, threshold in enumerate(thresholds):
        head_match_rows, head_match_columns = get_matches(head_matches, threshold)
        confidence = int(confidences[idx])

        matched_pixels = set(zip(head_match_rows, head_match_columns))
        newly_matched_pixels = matched_pixels.difference(already_matched_pixels)

        for row, col in newly_matched_pixels:
            scratchpad.rectangle([col - 10, row - 10, col + 10, row + 10], fill=None, outline="red")
            staves_on_row = np.where(trebel_staves == row)[0]
            detected_pitch = "-"
            if len(staves_on_row) != 0:
                stave_row_number = staves_on_row[0] % 5
                scratchpad.text((col - 22, row - 8), treble_notes_on_line[stave_row_number], "red", font)
                detected_pitch = treble_notes_on_line[stave_row_number]

            detections.append(f"{row - 10} \t {col - 10} \t 10 \t 10 \t filled_note \t {detected_pitch} \t {confidence}")

        already_matched_pixels.update(matched_pixels)

    for row, col in zip(rest_1_row, rest_1_column):
        if 10 < row < image_data.shape[0]-10:
            scratchpad.rectangle([col - 10, row - 10, col + 10, row + 10], fill=None, outline="green")
            detections.append(f"{row-10} \t {col-10} \t 10 \t 10 \t eight_rest \t _")
    for row, col in zip(rest_2_row, rest_2_column):
        if 10 < row < image_data.shape[0]-10:
            scratchpad.rectangle([col - 10, row - 10, col + 10, row + 10], fill=None, outline="blue")
            detections.append(f"{row-10} \t {col-10} \t 10 \t 10 \t quarter_rest \t _")

    detected_image.save(f"detected.png")
    with open(f"detected.txt",'w') as f:
        f.write("\n".join(detections))
