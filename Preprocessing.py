from skimage.restoration import unwrap_phase
from skimage.io import imsave, imread
import numpy as np
import os
from photutils.psf import TukeyWindow
from pathlib import Path
import cv2
import main
import tifffile

#Wykonał Franciszek_Żuchowski
def parameters_pp(overlap, first_x, last_x, first_y, last_y, main_path,direction, file_name,LI,I):

    sign = -1

    cols_num = last_y - first_y + 1
    rows_num = last_x - first_x + 1
    paths=[]
    for row in range(rows_num):
        for col in range(cols_num):
            suffix=f'_x{row + first_x:0>3}_y{col + first_y:0>3}.tiff'
            full_path=main_path+'/'+file_name+suffix
            paths.append(full_path)
            print(full_path)

    params = {
        "main_path": main_path,
        "direction": direction,
        "file_name": file_name,
        "overlap": overlap,
        "rows_num": rows_num,
        "cols_num": cols_num,

        "sign": sign,
        "first_row":first_x,
        "last_row":last_x,
        "first_col":first_y,
        "last_col":last_y,
        "LI":LI,
        "I":I,
    }




    return paths, params,
def parameters_fft(first_x, last_x, first_y, last_y, main_path, file_name,NA,magnification,wavelenght,px_size):

    cols_num = last_y - first_y + 1
    rows_num = last_x - first_x + 1
    paths=[]
    for row in range(rows_num):
        for col in range(cols_num):
            suffix=f'_x{row + first_x:0>3}_y{col + first_y:0>3}.tiff'
            full_path=main_path+'/'+file_name+suffix
            paths.append(full_path)
            print(full_path)


    params_FFT = {
        "first_x": first_x,
        "last_x": last_x,
        "first_y": first_y,
        "last_y": last_y,
        "main_path": main_path,
        "file_name": file_name,
        "NA": NA,
        "magnification":magnification,
        "wavelenght": wavelenght,
        "px_size": px_size,
    }



    return paths, params_FFT
def AG_parameters(file_name, location, method):
    selected_method=1
    if method == 'One-side cropping':
        selected_method=1
    elif method == 'Two-side cropping':
        selected_method=2
    elif method == 'Linear blending':
        selected_method=3
    elif method == 'Linear blending v2':
        selected_method=4
    elif method == 'Linear blending for full images':
        selected_method=5

    return file_name, location, selected_method

class Fourier_Transform:
    def __init__(self,height,width,binning=1,NA=0.45,wavelength=0.633,px_size=3.45,magnification=16):


        self.height = height
        self.width = width
        self.binning = binning
        self.NA= NA
        self.wavelength= wavelength
        self.px_size = px_size
        self.magnification = magnification

        self.holo = np.ones([self.height, self.width], dtype=np.complex64)

        self.y = (np.arange(self.height) - self.height // 2) / (self.height / 2)
        self.x = (np.arange(self.width) - self.width // 2) / (self.width / 2)
        self.yy, self.xx = np.meshgrid(self.y, self.x, indexing='ij')
        self.xx = self.xx.astype(np.complex64)
        self.yy = self.yy.astype(np.complex64)
        self.xx_gpu = np.asarray(self.xx)
        self.yy_gpu = np.asarray(self.yy)

    def Tukey_Window(self):#funkcja do apodyzacji pierwszego obrazu, ale źle działa to nieużywana
        tukey_window=TukeyWindow(alpha=0.7)
        self.apodization_filter=tukey_window((self.height,self.width))

    def Tukey_Window2(self):#apodyzacja wyciętego piku informacyjnego
        tukey_window = TukeyWindow(alpha=1)
        self.apodization_filter2 = tukey_window((self.diam_y, self.diam_x))
    def Get_Spectrum(self):#spectrum z pierwotnego hologramu
        if self.holo.dtype != np.complex64:
            self.holo = self.holo.astype(np.complex64)

        self.holo_after_apodization = np.asarray(self.holo)
        #self.holo_after_apodization *= self.apodization_filter
        self.spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(self.holo_after_apodization)))
        return self.spectrum
    def Detect_Information_Peak(self): #detekca piku informacyjnego
        print(self.holo.shape)
        Ny, Nx = self.holo.shape

        self.crop_spectrum = np.abs(self.spectrum)
        self.crop_spectrum[Ny // 2 - Ny // 100: Ny // 2 + Ny // 100, Nx // 2 - Nx // 100: Nx // 2 + Nx // 100] = 0
        self.crop_spectrum[:Ny // 2 + 1, :] = 0
        self.information_peak_y, self.information_peak_x = np.unravel_index(np.argmax(self.crop_spectrum), self.holo.shape)
        n = 7
        y = np.arange(2 * n + 1) + self.information_peak_y - n
        x = np.arange(2 * n + 1) + self.information_peak_x - n
        self.P = np.abs(self.crop_spectrum[np.ix_(y, x)]) ** 4
        self.information_peak_x = np.dot(np.sum(self.P, 0), x) / np.sum(self.P)
        self.information_peak_y = np.dot(np.sum(self.P, 1), y) / np.sum(self.P)
        print(self.information_peak_y, self.information_peak_x)
    def lincarrier2d_gpu(self): #nieużywana funkcja do jakiegoś wyrównania
        self.lincarr_gpu = np.exp(1j * np.pi * self.information_peak_y * self.yy_gpu) * np.exp(1j * np.pi * self.information_peak_x * self.xx_gpu)

    def Calculate_Information_Area(self): #funkcja pozwalająca na na obliczenie w zależności od parametrów mikroskopu informacji
        self.diam_x = np.int32(2 * np.round(self.NA * self.width * (self.px_size / self.magnification) / self.wavelength))
        self.diam_y = np.int32(2 * np.round(self.NA * self.height * (self.px_size / self.magnification)/ self.wavelength))
        top_y = int(self.information_peak_y) - ((self.diam_y) // 2)
        top_x = int(self.information_peak_x) - ((self.diam_x) // 2)
        self.information_spectrum = self.spectrum[top_y: top_y + self.diam_y, top_x: top_x + self.diam_x]
        print(self.information_spectrum.shape)
    def IFFT2(self):
        second_apodization =self.information_spectrum* self.apodization_filter2
        wrapped_phase = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(second_apodization)))
        wrapped_phase= wrapped_phase/ self.d_field_ref
        x=unwrap_phase(np.angle(wrapped_phase))
        #x-=x.mean()
        return wrapped_phase
    def ph_ref(self, sign=True):
        sgn = 0 if sign else np.pi
        self.ph = np.ones((self.diam_y, self.diam_x), np.float32)
        self.field_ref = np.exp(1j * (self.ph - sgn))
        self.ref = unwrap_phase(np.angle(self.field_ref)[
                                10:self.diam_y-10, 10:self.diam_x-10])
        self.d_field_ref = self.field_ref

class ImageCollection:
    def __init__(self, paths, params):
        self.params = params
        self.paths = paths
        self.direction = params['direction']
        self.main_path = params['main_path']
        self.first_row = params['first_row']
        self.last_row = params['last_row']
        self.first_col = params['first_col']
        self.last_col = params['last_col']
        self.LI = params['LI']
        self.I = params['I']
        self.after_abr=[]
        self.after_gradient=[]
        self.after_bc=[]
        self.frames_paths=paths

        if 'sign' in params:
            self.sign = params['sign']
        else:
            self.sign = 1
        if 'overlap' in params:
            self.overlap = params['overlap']
        else:
            self.overlap = 0
        if 'rows' in params:
            self.rows = params['rows_num']
        else:
            self.rows = 1
        if 'cols' in params:
            self.cols = params['cols_num']
        else:
            self.cols = 1
        self.mean_aberr = np.zeros((self.rows, self.cols))
        self.offsets = np.zeros(self.cols * self.rows)
        self.planes_params = np.zeros(self.cols * self.rows * 3)
        if 'px_size' in params:
            self.px_size = params['px_size']
        else:
            self.px_size = {'x': 1, 'y': 1}
        if 'description' in params:
            self.description = params['description']
        else:
            self.description = ''

        self.abr_path = Path(os.path.basename(self.paths[0][0])[:16] + '.tiff')
        self.abr_path = Path(os.path.dirname(self.paths[0][0]) / self.abr_path)

        self.frames = [[] for _ in range(self.rows)]
        counter=0
        for row in range(self.rows):
            for col in range(self.cols):
                image = imread(paths[counter])[0]  # odczytaj obraz z pliku
                self.frames[row].append(image)
                counter +=1
        self.name = params['file_name']
        self.file_name = []
        for row in range(self.first_row, self.last_row+1):
            for col in range(self.first_col, self.last_col+1):
                name2 = f'_x{row:0>3}_y{col :0>3}.tiff'
                full_name = self.name + name2
                self.file_name.append(full_name)
        im = imread(str(self.paths[0]))

        self.table_with_im = [[0 for x in range(self.last_row)] for y in range(self.last_col)]

    #wyliczenie wartości błędu systematycznego jako macierzy o wymiarach pojedynczego obrazu w sekwencji
    def return_average_aberrations(self):
        def recursive_mean(additional_data, mean_so_far, iteration):
            if iteration == 1:
                mean_so_far = additional_data
            else:
                mean_so_far = (iteration - 1) / iteration * mean_so_far + (additional_data / iteration) #wzór na część wspólną wszystkich obrazów w sekwencji

            return mean_so_far

        im = self.table_with_im[0][0]
        self.height = im.shape[0]
        self.width = im.shape[1]

        abr = np.zeros_like(im)
        iteration = 1 #ilość iteracji z których zbierane jest uśrednienie, można modyfikować
        i = 0
        for row in range(self.first_row, self.last_row + 1):
            for col in range(self.first_col, self.last_col + 1):
                im = self.table_with_im[col-1][row-1]
                abr = recursive_mean(im, abr, iteration)
                i += 1
                iteration += 1
        #tifffile.imsave(str("C:/Users/frane/OneDrive/Desktop/test_poprawki/A.tiff"), abr)
        return abr #macierz błędu systematycznego o wymiarach pojedynczego obrazu

    #odjęcie błędu systematycznego od kolejnych obrazów
    def remove_precalc_average_aberrations(self, abr):
        i=0
        imsave(str(self.main_path + '/Systematic_aberrtions.tiff'), abr)
        #tifffile.imsave(str("C:/Users/frane/OneDrive/Desktop/test_poprawki/AAAAA.tiff"), abr)

        reverse_counter=self.last_col
        for row in range(self.first_row, self.last_row + 1):
            for col in range(self.first_col, self.last_col + 1):
                postfix = f'_x{row :0>3}_y{col :0>3}.tiff'
                im = self.table_with_im[col-1][row-1]
                im_removed = im - 0.99*abr
                self.table_with_im[col-1][row-1]=im_removed
                #tifffile.imsave(str("C:/Users/frane/OneDrive/Desktop/test_poprawki/BEZ_ABR"+postfix), im_removed)

                self.after_abr.append(im_removed)
                i+=1
        # self.description += '_AME'

    #wyliczenie i usunięcie pochyłu fazy
    def gradient_slopes(self, sigma=50):
        from scipy.ndimage import gaussian_filter, median_filter

        def filter_gradient(grad):
            grad_filt = gaussian_filter(grad, sigma=sigma)
            slope = np.median(grad_filt)
            return slope

        def convert_to_2d(im):
            if len(im.shape) == 3:
                # obliczenie sumy dla każdego pixela dla 15 warstw
                sum_im = np.mean(im, axis=0)
                #print('wartosc', sum_im)
                return sum_im
            else:
                return im

        y = np.arange(self.height) - self.height // 2
        x = np.arange(self.width) - self.width // 2
        x, y = np.meshgrid(x, y)
        counter=0
        c=0
        for row in range(self.first_row, self.last_row+1):
            for col in range(self.first_col, self.last_col+1):

                postfix = f'_x{row :0>3}_y{col :0>3}.tiff'
                im=self.table_with_im[col-1][row-1]
                counter += 1
                print(im.shape)
                grad_y, grad_x = np.gradient(im)
                y_slope = filter_gradient(grad_y)
                im = np.array(im, dtype=np.float64)
                im -= y_slope *0.5*y
                x_slope = filter_gradient(grad_x)
                im -= x_slope * 0.5*x
                #tifffile.imsave(str("C:/Users/frane/OneDrive/Desktop/test_poprawki/gradient" + postfix), im)
                self.table_with_im[col-1][row-1]=im
        self.description += '_gradPF'
    #ustalenie poziomu tła
    def baseline_correction(self):

        first_img = self.table_with_im[0][0]
        overlap_y = self.overlap
        overlap_x = self.overlap
        print(overlap_y,overlap_x)
        y = first_img.shape[0]
        x = first_img.shape[1]
        margin_y = int(y * overlap_y)
        margin_x = int(x * overlap_x)

        array_of_frames = [[0 for x in range(self.last_row)] for y in range(self.last_col)]
        correction_array = [[0 for x in range(self.last_row)] for y in range(self.last_col)]
        offset = [[0 for x in range(self.last_row)] for y in range(self.last_col)]
        counter=0

        #stworzenie tablicy obrazów modyfikowanych
        for col in range(self.first_col-1, self.last_col):
            for row in range(self.first_row-1, self.last_row):
                current_image=self.table_with_im[col][row]
                counter+=1
                array_of_frames[col][row] = current_image

        #iteracja wyrównywania tła
        for iteration in range(self.I):
            print(iteration)

            #ALGORYTM OBLICZENIOWY DLA CAŁEGO ZAKRESU SEKWENCJI
            for row in range(self.last_row):
                for col in range(self.last_col):
                    if (col == 0 and (row > 0 and row < self.last_row - 1)):
                        C = array_of_frames[col][row]
                        B = array_of_frames[col + 1][row]
                        diff_B = -B[0:margin_y, :] + C[y - margin_y:y, :]
                        E_B = np.mean((np.sqrt((-B[0:margin_y, :] + C[y - margin_y:y, :]) ** 2) * np.sign(diff_B)))
                        L = array_of_frames[col][row - 1]
                        diff_L = -L[:, x - margin_x:x] + C[:, 0:margin_x]
                        E_L = np.mean((np.sqrt((-L[:, x - margin_x:x] + C[:, 0:margin_x]) ** 2) * np.sign(diff_L)))
                        R = array_of_frames[col][row + 1]
                        diff_R = -R[:, 0:margin_x] + C[:, x - margin_x:x]
                        E_R = np.mean((np.sqrt((-R[:, 0:margin_x] + C[:, x - margin_x:x]) ** 2) * np.sign(diff_R)))
                        P = self.LI * (E_L + E_B + E_R) / 3
                        correction_array[col][row] = P
                    elif (row == 0 and (col > 0 and col < self.last_col - 1)):
                        C = array_of_frames[col][row]
                        T = array_of_frames[col - 1][row]
                        diff_T = -T[y - margin_y:y, :] + C[0:margin_y, :]
                        E_T = np.mean((np.sqrt((-T[y - margin_y:y, :] + C[0:margin_y, :]) ** 2) * np.sign(diff_T)))
                        B = array_of_frames[col + 1][row]
                        diff_B = -B[0:margin_y, :] + C[y - margin_y:y, :]
                        E_B = np.mean((np.sqrt((-B[0:margin_y, :] + C[y - margin_y:y, :]) ** 2) * np.sign(diff_B)))
                        R = array_of_frames[col][row + 1]
                        diff_R = -R[:, 0:margin_x] + C[:, x - margin_x:x]
                        E_R = np.mean((np.sqrt((-R[:, 0:margin_x] + C[:, x - margin_x:x]) ** 2) * np.sign(diff_R)))
                        P = self.LI * (E_B +E_R + E_T) / 3
                        correction_array[col][row] = P
                    elif (row == self.last_row - 1 and (col > 0 and col < self.last_col - 1)):
                        C = array_of_frames[col][row]
                        T = array_of_frames[col - 1][row]
                        diff_T = -T[y - margin_y:y, :] + C[0:margin_y, :]
                        E_T = np.mean((np.sqrt((-T[y - margin_y:y, :] + C[0:margin_y, :]) ** 2) * np.sign(diff_T)))
                        B = array_of_frames[col + 1][row]
                        diff_B = -B[0:margin_y, :] + C[y - margin_y:y, :]
                        E_B = np.mean((np.sqrt((-B[0:margin_y, :] + C[y - margin_y:y, :]) ** 2) * np.sign(diff_B)))
                        L = array_of_frames[col][row - 1]
                        diff_L = -L[:, x - margin_x:x] + C[:, 0:margin_x]
                        E_L = np.mean((np.sqrt((-L[:, x - margin_x:x] + C[:, 0:margin_x]) ** 2) * np.sign(diff_L)))
                        P = self.LI * (E_L+E_B+E_T) / 3
                        correction_array[col][row] = P
                    elif (col == self.last_col - 1 and (row > 0 and row < self.last_row - 1)):
                        T = array_of_frames[col - 1][row]
                        L = array_of_frames[col][row - 1]
                        R = array_of_frames[col][row + 1]
                        C = array_of_frames[col][row]
                        diff_L = -L[:, x - margin_x:x] + C[:, 0:margin_x]
                        diff_R = -R[:, 0:margin_x] + C[:, x - margin_x:x]
                        diff_T = -T[y - margin_y:y, :] + C[0:margin_y, :]
                        E_T = np.mean((np.sqrt((-T[y - margin_y:y, :] + C[0:margin_y, :]) ** 2) * np.sign(diff_T)))
                        E_L = np.mean((np.sqrt((-L[:, x - margin_x:x] + C[:, 0:margin_x]) ** 2) * np.sign(diff_L)))
                        E_R = np.mean((np.sqrt((-R[:, 0:margin_x] + C[:, x - margin_x:x]) ** 2) * np.sign(diff_R)))
                        P = self.LI * ((np.mean(E_L) + np.mean(E_T) + np.mean(E_R)) / 3)
                        correction_array[col][row] = P
                    elif (row > 0 and row < self.last_row - 1) and (col > 0 and col < self.last_col - 1):
                        T = array_of_frames[col - 1][row]
                        B = array_of_frames[col + 1][row]
                        L = array_of_frames[col][row - 1]
                        R = array_of_frames[col][row + 1]
                        C = array_of_frames[col][row]
                        diff_T = -T[y - margin_y:y, :] + C[0:margin_y, :]
                        diff_B = -B[0:margin_y, :] + C[y - margin_y:y, :]
                        diff_L = -L[:, x - margin_x:x] + C[:, 0:margin_x]
                        diff_R = -R[:, 0:margin_x] + C[:, x - margin_x:x]
                        E_T = np.mean((np.sqrt((-T[y - margin_y:y, :] + C[0:margin_y, :]) ** 2) * np.sign(diff_T)))
                        E_B = np.mean((np.sqrt((-B[0:margin_y, :] + C[y - margin_y:y, :]) ** 2) * np.sign(diff_B)))
                        E_L = np.mean((np.sqrt((-L[:, x - margin_x:x] + C[:, 0:margin_x]) ** 2) * np.sign(diff_L)))
                        E_R = np.mean((np.sqrt((-R[:, 0:margin_x] + C[:, x - margin_x:x]) ** 2) * np.sign(diff_R)))
                        P = self.LI * (E_R + E_L + E_B + E_T) / 4
                        correction_array[col][row] = P
                    elif row == 0 and col == 0:
                        B = array_of_frames[col + 1][row]
                        R = array_of_frames[col][row + 1]
                        C = array_of_frames[col][row]
                        diff_B = -B[0:margin_y, :] + C[y - margin_y:y, :]
                        diff_R = -R[:, 0:margin_x] + C[:, x - margin_x:x]
                        E_B = np.mean((np.sqrt((-B[0:margin_y, :] + C[y - margin_y:y, :]) ** 2) * np.sign(diff_B)))
                        E_R = np.mean((np.sqrt((-R[:, 0:margin_x] + C[:, x - margin_x:x]) ** 2) * np.sign(diff_R)))
                        P = self.LI * (E_B + E_R) / 2
                        correction_array[col][row] = P
                    elif row == self.last_row - 1 and col == self.last_col - 1:
                        T = array_of_frames[col - 1][row]
                        L = array_of_frames[col][row - 1]
                        C = array_of_frames[col][row]
                        diff_L = -L[:, x - margin_x:x] + C[:, 0:margin_x]
                        diff_T = -T[y - margin_y:y, :] + C[0:margin_y, :]
                        E_T = np.mean((np.sqrt((-T[y - margin_y:y, :] + C[0:margin_y, :]) ** 2) * np.sign(diff_T)))
                        E_L = np.mean((np.sqrt((-L[:, x - margin_x:x] + C[:, 0:margin_x]) ** 2) * np.sign(diff_L)))
                        P = self.LI * (E_L + E_T) / 2
                        correction_array[col][row] = P
                    elif row == 0 and col == self.last_col - 1:
                        T = array_of_frames[col - 1][row]
                        R = array_of_frames[col][row + 1]
                        C = array_of_frames[col][row]
                        diff_T = -T[y - margin_y:y, :] + C[0:margin_y, :]
                        diff_R = -R[:, 0:margin_x] + C[:, x - margin_x:x]
                        E_T = np.mean((np.sqrt((-T[y - margin_y:y, :] + C[0:margin_y, :]) ** 2) * np.sign(diff_T)))
                        E_R = np.mean((np.sqrt((-R[:, 0:margin_x] + C[:, x - margin_x:x]) ** 2) * np.sign(diff_R)))
                        P = self.LI * (E_T + E_R) / 2
                        correction_array[col][row] = P
                    elif row == self.last_row - 1 and col == 0:
                        B = array_of_frames[col + 1][row]
                        L = array_of_frames[col][row - 1]
                        C = array_of_frames[col][row]
                        diff_B = -B[0:margin_y, :] + C[y - margin_y:y, :]
                        diff_L = -L[:, x - margin_x:x] + C[:, 0:margin_x]
                        E_B = np.mean((np.sqrt((-B[0:margin_y, :] + C[y - margin_y:y, :]) ** 2) * np.sign(diff_B)))
                        E_L = np.mean((np.sqrt((-L[:, x - margin_x:x] + C[:, 0:margin_x]) ** 2) * np.sign(diff_L)))
                        P = self.LI * (E_L + E_B) / 2
                        correction_array[col][row] = P

            for i in range(self.last_row):
                for j in range(self.last_col):
                    X = correction_array[j][i]
                    offset[j][i] = X * np.ones_like(array_of_frames[j][i])
                    print(X)
            j = 0
            i = 0
            array_of_frames = np.array(array_of_frames)

            array_of_frames = array_of_frames.astype('float64')
            for j in range(self.last_col):
                for i in range(self.last_row):
                    array_of_frames[j][i] -= offset[j][i] #odjęcie offsetu


            print('Counter: ',iteration)


        for row in range(self.first_row,self.last_row+1):
            for col in range(self.first_col, self.last_col + 1):
                reverse_counter = self.last_col-col+1
                postfix = f'_x{row :0>3}_y{col :0>3}.tiff'
                tifffile.imsave(str("C:/Users/frane/OneDrive/Desktop/test_poprawki/GOTOWE" + postfix), array_of_frames[col - 1][row - 1])
                name2 = "/After_preproc"+  f'_x{row:0>3}_y{col :0>3}.tiff'
                imsave((self.direction + name2), array_of_frames[col - 1][row - 1])
        return correction_array
    #metoda one-side cropping
    def arrange_grid(self,file_name,location,overlap,first_y,first_x,last_y,last_x):

        first_img = imread(str(location+'/'+file_name+"_x001_y001.tiff"))
        y = first_img.shape[0]
        x = first_img.shape[1]
        overlap_y = overlap
        overlap_x = overlap
        y_frames = last_y - first_y + 1
        x_frames = last_x - first_x + 1
        margin_y = int(y * overlap_y)
        margin_x = int(x * overlap_x)
        if margin_x%2!=0:
            margin_x+=1
        if margin_y%2!=0:
            margin_y+=1
        x_size = int((y - margin_y) * y_frames + margin_y)
        y_size = int((x - margin_x) * x_frames + margin_x)
        print(y_size, x_size)
        full_image = np.ones((x_size, y_size))

        for row in range(last_x):
            for col in range(last_y):

                suffix =  f'_x{row+1:0>3}_y{col+1 :0>3}.tiff'
                current_image = imread(str(location+'/'+file_name+suffix))
                x_start = int((row)) * (x - margin_x)
                y_start = int((col)) * (y - margin_y)
                x_end = (row+1) * (x - margin_x)
                y_end = (col+1) * (y - margin_y)
                if row == last_x-1 and col != last_y-1:
                    crop_image = current_image[0:y - margin_y, 0:x]
                    full_image[y_start:y_end, x_start:x_end + margin_x] = crop_image
                elif col == last_y-1 and row != last_x-1:
                    crop_image = current_image[0:y, 0:x - margin_x]
                    full_image[y_start:y_end + margin_y, x_start:x_end] = crop_image
                elif col == last_y-1 and row == last_x-1:
                    crop_image = current_image[0:y, 0:x]
                    full_image[y_start:y_end + margin_y, x_start:x_end + margin_x] = crop_image
                else:
                    crop_image = current_image[0:y - margin_y, 0:x - margin_x]
                    full_image[y_start:y_end, x_start:x_end] = crop_image

        return full_image

    #metoda two-side cropping
    def arrange_grid2(self,file_name,location,overlap,first_y,first_x,last_y,last_x):

        first_img = imread(str(location+'/'+file_name+'_x001_y001.tiff'))
        y = first_img.shape[0]
        x = first_img.shape[1]
        overlap_y = overlap
        overlap_x = overlap
        y_frames = last_y - first_y + 1
        x_frames = last_x - first_x + 1
        margin_y = int(y * overlap_y)
        margin_x = int(x * overlap_x)
        if margin_x%2!=0:
            margin_x+=1
        if margin_y%2!=0:
            margin_y+=1
        x_size = int((y - margin_y) * y_frames + margin_y)
        y_size = int((x - margin_x) * x_frames + margin_x)
        print(y_size, x_size)
        full_image = np.ones((x_size, y_size))

        for row in range(last_x):
            for col in range(last_y):
                #reverse_counter = last_y + 1 - col
                suffix = f'_x{row+1:0>3}_y{col+1 :0>3}.tiff'
                current_image = imread(str(location+'/'+file_name+suffix))
                x_start = int((row)) * (x - margin_x)
                y_start = int((col)) * (y - margin_y)
                x_end = (row+1) * (x - margin_x)
                y_end = (col+1) * (y - margin_y)
                if row==0 and col==0:
                    crop_image = current_image[0:y - margin_y//2, 0:x-margin_x//2]
                    full_image[y_start:y_end+margin_y//2, x_start:x_end + margin_x//2] = crop_image
                elif (row>0 and row<last_x-1) and col == 0:
                    crop_image = current_image[0:y - margin_y//2, margin_x//2:x-margin_x//2]
                    full_image[y_start:y_end+margin_y//2, x_start+margin_x//2:x_end + margin_x//2] = crop_image
                elif row==last_x-1 and col==0:
                    crop_image = current_image[0:y-margin_y//2, margin_x//2:x]
                    full_image[y_start:y_end + margin_y//2, x_start+margin_x//2:x_end+margin_x] = crop_image
                elif (col >0 and col <last_y-1) and row == 0:
                    crop_image = current_image[margin_y//2:y-margin_y//2, 0:x-margin_x//2]
                    full_image[y_start+margin_y//2:y_end+margin_y//2, x_start:x_end + margin_x//2] = crop_image
                elif col==last_y-1 and row==0:
                    crop_image = current_image[margin_y//2:y, 0:x - margin_x//2]
                    full_image[y_start+margin_y//2:y_end+margin_y, x_start:x_end+margin_x//2] = crop_image
                elif (col>0 and col<last_y-1) and (row>0 and row < last_x-1):
                    crop_image = current_image[margin_y//2:y-margin_y//2, margin_x//2:x - margin_x//2]
                    full_image[y_start+margin_y//2:y_end+margin_y//2, x_start+margin_x//2:x_end+margin_x//2] = crop_image
                elif col==last_y-1 and row>0 and (row < last_x-1):
                    crop_image = current_image[margin_y//2:y, margin_x//2:x - margin_x//2]
                    full_image[y_start+margin_y//2:y_end+margin_y, x_start+margin_x//2:x_end+margin_x//2] = crop_image
                elif (col>0 and col<last_y-1) and row==last_x-1:
                    crop_image = current_image[margin_y//2:y-margin_y//2, margin_x//2:x]
                    full_image[y_start+margin_y//2:y_end+margin_y//2, x_start+margin_x//2:x_end+margin_x] = crop_image
                elif col==last_y-1 and row==last_x-1:
                    crop_image = current_image[margin_y // 2:y, margin_x // 2:x]
                    full_image[y_start + margin_y // 2:y_end + margin_y,x_start + margin_x // 2:x_end + margin_x] = crop_image

        return full_image

    #metoda linear blending
    def arrange_grid3(self,file_name,location,overlap,first_y,first_x,last_y,last_x):
        first_img = imread(str(location+'/'+file_name+'_x001_y001.tiff'))
        y = first_img.shape[0]
        x = first_img.shape[1]
        overlap_y = overlap
        overlap_x = overlap
        y_frames = last_y - first_y + 1
        x_frames = last_x - first_x + 1
        margin_y = int(y * overlap_y)
        margin_x = int(x * overlap_x)
        if margin_x%2!=0:
            margin_x+=1
        if margin_y%2!=0:
            margin_y+=1
        x_size = int((y - margin_y) * y_frames + margin_y)
        y_size = int((x - margin_x) * x_frames + margin_x)
        print(y_size, x_size)
        full_image = np.ones((x_size, y_size))
        array_of_images = [[0 for i in range(last_x)] for j in range(last_y)]
        for row in range(last_x):
            for col in range(last_y):
                suffix = f'_x{row+1:0>3}_y{col+1 :0>3}.tiff'
                array_of_images[col][row] = imread(str(location + '/' + file_name + suffix))
        for row in range(first_x, last_x + 1):
            for col in range(first_y, last_y + 1):
                current_img = array_of_images[col - 1][row - 1]

                x_start = int((row - 1)) * (x - margin_x)
                y_start = int((col - 1)) * (y - margin_y)
                x_end = row * (x - margin_x)
                y_end = col * (y - margin_y)
                # NAROŻNIKI
                if (col == first_y) and (row == first_x):  # warunek dla zdjęcia x001y001 (lewy gorny)
                    bottom_img = array_of_images[col][row - 1]
                    right_img = array_of_images[col - 1][row]
                    y_margin_TOP = bottom_img[0:margin_y, 0:x - margin_x]
                    y_margin_BOT = current_img[y - margin_y:y, 0:x - margin_x]
                    x_margin_RIGHT = current_img[0:y, x - margin_x:x]
                    x_margin_LEFT = right_img[0:y, 0:margin_x]
                    down_overlap = cv2.addWeighted(y_margin_TOP, 0.5, y_margin_BOT, 0.5, 0)
                    right_overlap = cv2.addWeighted(x_margin_LEFT, 0.5, x_margin_RIGHT, 0.5, 0)
                    crop_image = current_img[0:y - margin_y, 0:x - margin_x]
                    full_image[y_start:y_end, 0:x_end] = crop_image
                    full_image[y_end:y_end + margin_y, 0:x_end] = down_overlap
                    full_image[0:y_end + margin_y, x_end:x_end + margin_x] = right_overlap
                    print('lewy górny')
                elif (col == last_y) and (row == first_x):
                    right_img = array_of_images[col - 1][row]
                    x_margin_RIGHT = current_img[margin_y:y, x - margin_x:x]
                    x_margin_LEFT = right_img[margin_y:y, 0:margin_x]
                    right_overlap = cv2.addWeighted(x_margin_LEFT, 0.5, x_margin_RIGHT, 0.5, 0)
                    crop_image = current_img[margin_y:y, margin_x:x]
                    full_image[y_start + margin_y:y_end + margin_y, 0:x_end] = crop_image
                    full_image[y_start + margin_y:y_end + margin_y, x_end:x_end + margin_x] = right_overlap
                    print('lewy dolny')
                elif (col == first_y) and (row == last_x):
                    bottom_img = array_of_images[col][row - 1]
                    y_margin_TOP = bottom_img[0:margin_y, 0:x - margin_x]
                    y_margin_BOT = current_img[y - margin_y:y, 0:x - margin_x]
                    down_overlap = cv2.addWeighted(y_margin_TOP, 0.5, y_margin_BOT, 0.5, 0)
                    crop_image = current_img[0:y - margin_y, margin_x:x]
                    full_image[y_start:y_end, margin_x + x_start:x_end + margin_x] = crop_image
                    full_image[y_end:y_end + margin_y, margin_x + x_start:x_end + margin_x] = down_overlap
                    print('prawy gorny')
                elif (col == last_y) and (row == last_x):
                    crop_image = current_img[margin_y:y, margin_x:x]
                    full_image[y_start + margin_y:y_end + margin_y, x_start + margin_x:x_end + margin_x] = crop_image
                    print('prawy dolny')
                # KRAWĘDZIE
                elif col == first_y and (row > first_x and row < last_x):
                    current_img = array_of_images[col - 1][row - 1]
                    bottom_img = array_of_images[col][row - 1]
                    right_img = array_of_images[col - 1][row]
                    y_margin_TOP = bottom_img[0:margin_y, margin_x:x]
                    y_margin_BOT = current_img[y - margin_y:y, margin_x:x]
                    x_margin_RIGHT = current_img[0:y - margin_y, x - margin_x:x]
                    x_margin_LEFT = right_img[0:y - margin_y, 0:margin_x]
                    down_overlap = cv2.addWeighted(y_margin_TOP, 0.5, y_margin_BOT, 0.5, 0)
                    right_overlap = cv2.addWeighted(x_margin_LEFT, 0.5, x_margin_RIGHT, 0.5, 0)
                    crop_image = current_img[0:y - margin_y, margin_x:x - margin_x]
                    full_image[y_start:y_end, x_start + margin_x:x_end] = crop_image
                    full_image[y_end:y_end + margin_y, x_start + margin_x:x_end + margin_x] = down_overlap
                    full_image[y_start:y_end, x_end:x_end + margin_x] = right_overlap
                    print('pierwszy rząd')
                elif row == first_x and (col > first_y and col < last_y):
                    bottom_img = array_of_images[col][row - 1]
                    right_img = array_of_images[col - 1][row]
                    y_margin_TOP = bottom_img[0:margin_y, 0:x - margin_x]
                    y_margin_BOT = current_img[y - margin_y:y, 0:x - margin_x]
                    x_margin_RIGHT = current_img[0:y, x - margin_x:x]
                    x_margin_LEFT = right_img[0:y, 0:margin_x]
                    down_overlap = cv2.addWeighted(y_margin_TOP, 0.5, y_margin_BOT, 0.5, 0)
                    right_overlap = cv2.addWeighted(x_margin_LEFT, 0.5, x_margin_RIGHT, 0.5, 0)
                    crop_image = current_img[margin_y:y - margin_y, 0:x - margin_x]
                    full_image[y_start + margin_y:y_end, x_start:x_end] = crop_image
                    full_image[y_end:y_end + margin_y, x_start:x_end] = down_overlap
                    full_image[y_start:y_end + margin_y, x_end:x_end + margin_x] = right_overlap
                    print('pierwsza kolumna')
                elif col == last_y and (row > first_x and row < last_x):
                    right_img = array_of_images[col - 1][row]
                    x_margin_RIGHT = current_img[0:y, x - margin_x:x]
                    x_margin_LEFT = right_img[0:y, 0:margin_x]
                    right_overlap = cv2.addWeighted(x_margin_LEFT, 0.5, x_margin_RIGHT, 0.5, 0)
                    crop_image = current_img[margin_y:y, margin_x:x - margin_x]
                    full_image[y_start + margin_y:y_end + margin_y, x_start + margin_x:x_end] = crop_image
                    full_image[y_start:y_end + margin_y, x_end:x_end + margin_x] = right_overlap
                    print('dolny rzad')
                elif row == last_x and (col > first_y and col < last_y):
                    bottom_img = array_of_images[col][row - 1]
                    y_margin_TOP = bottom_img[0:margin_y, margin_x:x]
                    y_margin_BOT = current_img[y - margin_y:y, margin_x:x]
                    down_overlap = cv2.addWeighted(y_margin_TOP, 0.5, y_margin_BOT, 0.5, 0)
                    crop_image = current_img[margin_y:y - margin_y, margin_x:x]
                    full_image[y_start + margin_y:y_end, x_start + margin_x:x_end + margin_x] = crop_image
                    full_image[y_end:y_end + margin_y, x_start + margin_x:x_end + margin_x] = down_overlap
                    print('ostatnia kolumna')
                # ŚRODEK
                elif (col > first_y and col < last_y) and (row > first_x and row < last_x):
                    bottom_img = array_of_images[col][row - 1]
                    right_img = array_of_images[col - 1][row]

                    y_margin_TOP = bottom_img[0:margin_y, margin_x:x - margin_x].astype(np.float32)
                    y_margin_BOT = current_img[y - margin_y:y, margin_x:x - margin_x].astype(np.float32)
                    x_margin_RIGHT = current_img[0:y, x - margin_x:x].astype(np.float32)
                    x_margin_LEFT = right_img[0:y, 0:margin_x].astype(np.float32)
                    down_overlap2 = cv2.addWeighted(y_margin_TOP, 0.5, y_margin_BOT, 0.5, 0)
                    right_overlap2 = cv2.addWeighted(x_margin_LEFT, 0.5, x_margin_RIGHT, 0.5, 0)
                    current_image = array_of_images[col - 1][row - 1]
                    crop_image = current_image[margin_y:y - margin_y, margin_x:x - margin_x]
                    full_image[y_start + margin_y:y_end, x_start + margin_x:x_end] = crop_image
                    full_image[y_start:y_end + margin_y, x_end:x_end + margin_x] = right_overlap2
                    full_image[y_end:y_end + margin_y, x_start + margin_x:x_end] = down_overlap2
                    print("środek")
        return full_image

    #metoda linear blending v2
    def arrange_grid4(self,file_name,location,overlap,first_y,first_x,last_y,last_x):
        first_img = imread(str(location+'/'+file_name+'_x001_y001.tiff'))
        y = first_img.shape[0]
        x = first_img.shape[1]
        overlap_y = overlap
        overlap_x = overlap

        y_frames = last_y - first_y + 1
        x_frames = last_x - first_x + 1
        margin_y = int(y * overlap_y)
        margin_x = int(x * overlap_x)
        if margin_x%2!=0:
            margin_x+=1
        if margin_y%2!=0:
            margin_y+=1
        x_size = int((y - margin_y) * y_frames + margin_y)
        y_size = int((x - margin_x) * x_frames + margin_x)
        print(y_size, x_size)
        full_image = np.ones((x_size, y_size))
        array_of_images = [[0 for i in range(last_x)] for j in range(last_y)]
        for row in range(last_x):
            for col in range(last_y):
                suffix = f'_x{row + 1:0>3}_y{col + 1 :0>3}.tiff'
                array_of_images[col][row] = imread(str(location + '/' + file_name + suffix))
        for row in range(first_x, last_x + 1):
            for col in range(first_y, last_y + 1):
                current_img = array_of_images[col - 1][row - 1]

                x_start = int((row - 1)) * (x - margin_x)
                y_start = int((col - 1)) * (y - margin_y)
                x_end = row * (x - margin_x)
                y_end = col * (y - margin_y)
                # NAROŻNIKI
                if (col == first_y) and (row == first_x):  # warunek dla zdjęcia x001y001 (lewy gorny)
                    bottom_img = array_of_images[col][row - 1]
                    right_img = array_of_images[col - 1][row]
                    bot_right = array_of_images[col][row]

                    y_margin_TOP = bottom_img[0:margin_y, 0:x - margin_x]
                    y_margin_BOT = current_img[y - margin_y:y, 0:x - margin_x]
                    x_margin_RIGHT = current_img[0:y - margin_y, x - margin_x:x]
                    x_margin_LEFT = right_img[0:y - margin_y, 0:margin_x]

                    down_overlap = cv2.addWeighted(y_margin_TOP, 0.5, y_margin_BOT, 0.5, 0)
                    right_overlap = cv2.addWeighted(x_margin_LEFT, 0.5, x_margin_RIGHT, 0.5, 0)
                    crop_image = current_img[0:y - margin_y, 0:x - margin_x]

                    ImageCollection.corner_calculate(self,current_img, right_img, bottom_img,
                                                     bot_right, full_image, margin_y, margin_x,x_end,y_end, x, y)
                    full_image[y_start:y_end, 0:x_end] = crop_image
                    full_image[y_end:y_end + margin_y, 0:x_end] = down_overlap
                    full_image[0:y_end, x_end:x_end + margin_x] = right_overlap

                    print('lewy górny')
                elif (col == last_y) and (row == first_x):
                    right_img = array_of_images[col - 1][row]
                    x_margin_RIGHT = current_img[margin_y:y, x - margin_x:x]
                    x_margin_LEFT = right_img[margin_y:y, 0:margin_x]
                    right_overlap = cv2.addWeighted(x_margin_LEFT, 0.5, x_margin_RIGHT, 0.5, 0)
                    crop_image = current_img[margin_y:y, margin_x:x]
                    full_image[y_start + margin_y:y_end + margin_y, 0:x_end] = crop_image
                    full_image[y_start + margin_y:y_end + margin_y, x_end:x_end + margin_x] = right_overlap
                    print('lewy dolny')
                elif (col == first_y) and (row == last_x):
                    bottom_img = array_of_images[col][row - 1]
                    y_margin_TOP = bottom_img[0:margin_y, 0:x - margin_x]
                    y_margin_BOT = current_img[y - margin_y:y, 0:x - margin_x]
                    down_overlap = cv2.addWeighted(y_margin_TOP, 0.5, y_margin_BOT, 0.5, 0)
                    crop_image = current_img[0:y - margin_y, margin_x:x]
                    full_image[y_start:y_end, margin_x + x_start:x_end + margin_x] = crop_image
                    full_image[y_end:y_end + margin_y, margin_x + x_start:x_end + margin_x] = down_overlap
                    print('prawy gorny')
                elif (col == last_y) and (row == last_x):
                    crop_image = current_img[margin_y:y, margin_x:x]
                    full_image[y_start + margin_y:y_end + margin_y, x_start + margin_x:x_end + margin_x] = crop_image
                    print('prawy dolny')
                # KRAWĘDZIE
                elif col == first_y and (row > first_x and row < last_x):
                    current_img = array_of_images[col - 1][row - 1]
                    bottom_img = array_of_images[col][row - 1]
                    right_img = array_of_images[col - 1][row]
                    bot_right = array_of_images[col][row]

                    y_margin_TOP = bottom_img[0:margin_y, margin_x:x - margin_x]
                    y_margin_BOT = current_img[y - margin_y:y, margin_x:x - margin_x]
                    x_margin_RIGHT = current_img[0:y - margin_y, x - margin_x:x]
                    x_margin_LEFT = right_img[0:y - margin_y, 0:margin_x]
                    ImageCollection.corner_calculate(self,current_img, right_img, bottom_img, bot_right, full_image, margin_y, margin_x,
                                     x_end, y_end, x, y)

                    down_overlap = cv2.addWeighted(y_margin_TOP, 0.5, y_margin_BOT, 0.5, 0)
                    right_overlap = cv2.addWeighted(x_margin_LEFT, 0.5, x_margin_RIGHT, 0.5, 0)
                    crop_image = current_img[0:y - margin_y, margin_x:x - margin_x]

                    full_image[y_start:y_end, x_start + margin_x:x_end] = crop_image
                    full_image[y_end:y_end + margin_y, x_start + margin_x:x_end] = down_overlap
                    full_image[y_start:y_end, x_end:x_end + margin_x] = right_overlap

                    print('pierwszy rząd')
                elif row == first_x and (col > first_y and col < last_y):
                    bottom_img = array_of_images[col][row - 1]
                    right_img = array_of_images[col - 1][row]
                    bot_right = array_of_images[col][row]

                    y_margin_TOP = bottom_img[0:margin_y, 0:x - margin_x]
                    y_margin_BOT = current_img[y - margin_y:y, 0:x - margin_x]
                    x_margin_RIGHT = current_img[margin_y:y - margin_y, x - margin_x:x]
                    x_margin_LEFT = right_img[margin_y:y - margin_y, 0:margin_x]

                    ImageCollection.corner_calculate(self,current_img, right_img, bottom_img, bot_right, full_image, margin_y, margin_x,
                                     x_end,
                                     y_end, x, y)
                    down_overlap = cv2.addWeighted(y_margin_TOP, 0.5, y_margin_BOT, 0.5, 0)
                    right_overlap = cv2.addWeighted(x_margin_LEFT, 0.5, x_margin_RIGHT, 0.5, 0)
                    crop_image = current_img[margin_y:y - margin_y, 0:x - margin_x]

                    full_image[y_start + margin_y:y_end, x_start:x_end] = crop_image
                    full_image[y_end:y_end + margin_y, x_start:x_end] = down_overlap
                    full_image[y_start + margin_y:y_end, x_end:x_end + margin_x] = right_overlap

                    print('pierwsza kolumna')
                elif col == last_y and (row > first_x and row < last_x):
                    right_img = array_of_images[col - 1][row]

                    x_margin_RIGHT = current_img[margin_y:y, x - margin_x:x]
                    x_margin_LEFT = right_img[margin_y:y, 0:margin_x]

                    right_overlap = cv2.addWeighted(x_margin_LEFT, 0.5, x_margin_RIGHT, 0.5, 0)
                    crop_image = current_img[margin_y:y, margin_x:x - margin_x]

                    full_image[y_start + margin_y:y_end + margin_y, x_start + margin_x:x_end] = crop_image
                    full_image[y_start + margin_y:y_end + margin_y, x_end:x_end + margin_x] = right_overlap
                    print('dolny rzad')
                elif row == last_x and (col > first_y and col < last_y):
                    bottom_img = array_of_images[col][row - 1]

                    y_margin_TOP = bottom_img[0:margin_y, margin_x:x]
                    y_margin_BOT = current_img[y - margin_y:y, margin_x:x]

                    down_overlap = cv2.addWeighted(y_margin_TOP, 0.5, y_margin_BOT, 0.5, 0)
                    crop_image = current_img[margin_y:y - margin_y, margin_x:x]

                    full_image[y_start + margin_y:y_end, x_start + margin_x:x_end + margin_x] = crop_image
                    full_image[y_end:y_end + margin_y, x_start + margin_x:x_end + margin_x] = down_overlap
                    print('ostatnia kolumna')
                # ŚRODEK
                elif (col > first_y and col < last_y) and (row > first_x and row < last_x):
                    bottom_img = array_of_images[col][row - 1]
                    right_img = array_of_images[col - 1][row]
                    bot_right = array_of_images[col][row]

                    y_margin_TOP = bottom_img[0:margin_y, margin_x:x - margin_x]
                    y_margin_BOT = current_img[y - margin_y:y, margin_x:x - margin_x]
                    x_margin_RIGHT = current_img[margin_y:y - margin_y, x - margin_x:x]
                    x_margin_LEFT = right_img[margin_y:y - margin_y, 0:margin_x]
                    ImageCollection.corner_calculate(self,current_img, right_img, bottom_img, bot_right, full_image, margin_y, margin_x,
                                     x_end, y_end, x, y)

                    down_overlap2 = cv2.addWeighted(y_margin_TOP, 0.5, y_margin_BOT, 0.5, 0)
                    right_overlap2 = cv2.addWeighted(x_margin_LEFT, 0.5, x_margin_RIGHT, 0.5, 0)
                    current_image = array_of_images[col - 1][row - 1]

                    crop_image = current_image[margin_y:y - margin_y, margin_x:x - margin_x]
                    full_image[y_start + margin_y:y_end, x_start + margin_x:x_end] = crop_image
                    full_image[y_start + margin_y:y_end, x_end:x_end + margin_x] = right_overlap2
                    full_image[y_end:y_end + margin_y, x_start + margin_x:x_end] = down_overlap2

                    print("środek")
        return full_image

    #funkcja pomocnicza dla linear blending for corners
    def corner_calculate(self,current_img, right_img, bottom_img, bot_right, full_image, margin_y, margin_x, x_end, y_end, x,y):
        corner_leftUP = current_img[y - margin_y:y, x - margin_x:x]
        corner_rightUP = right_img[y - margin_y:y, 0:margin_x]
        corner_leftDOWN = bottom_img[0:margin_y, x - margin_x:x]
        corner_rightDOWN = bot_right[0:margin_y, 0:margin_x]
        avg_1 = cv2.addWeighted(corner_leftUP, 0.5, corner_rightUP, 0.5, 0)
        avg_2 = cv2.addWeighted(corner_rightDOWN, 0.5, corner_leftDOWN, 0.5, 0)
        avg_3 = cv2.addWeighted(avg_1, 0.5, avg_2, 0.5, 0)




        full_image[y_end:y_end + margin_y, x_end:x_end + margin_x] = avg_3

    #linear blending for corners
    def arrange_grid5(self,file_name,location,overlap,first_y,first_x,last_y,last_x):
        first_img = imread(str(location + '/' + file_name + '_x001_y001.tiff'))
        y = first_img.shape[0]
        x = first_img.shape[1]
        overlap_y = overlap
        overlap_x = overlap

        y_frames = last_y - first_y + 1
        x_frames = last_x - first_x + 1
        margin_y = int(y * overlap_y)
        margin_x = int(x * overlap_x)
        if margin_x%2!=0:
            margin_x+=1
        if margin_y%2!=0:
            margin_y+=1
        x_size = int((y - margin_y) * y_frames + margin_y)
        y_size = int((x - margin_x) * x_frames + margin_x)
        print(y_size, x_size)
        full_image1 = np.ones((x_size, y_size))
        full_image2 = np.ones((x_size, y_size))

        for row in range(last_x):
            for col in range(last_y):
                #reverse_counter = last_y + 1 - col
                suffix = f'_x{row+1:0>3}_y{col+1 :0>3}.tiff'
                current_image = imread(str(location+'/'+file_name+suffix))
                x_start = int((row)) * (x - margin_x)
                y_start = int((col)) * (y - margin_y)
                x_end = (row+1) * (x - margin_x)
                y_end = (col+1) * (y - margin_y)
                if row==0 and col==0:
                    crop_image = current_image[0:y - margin_y//2, 0:x-margin_x//2]
                    full_image1[y_start:y_end+margin_y//2, x_start:x_end + margin_x//2] = crop_image
                elif (row>0 and row<last_x-1) and col == 0:
                    crop_image = current_image[0:y - margin_y//2, margin_x//2:x-margin_x//2]
                    full_image1[y_start:y_end+margin_y//2, x_start+margin_x//2:x_end + margin_x//2] = crop_image
                elif row==last_x-1 and col==0:
                    crop_image = current_image[0:y-margin_y//2, margin_x//2:x]
                    full_image1[y_start:y_end + margin_y//2, x_start+margin_x//2:x_end+margin_x] = crop_image
                elif (col >0 and col <last_y-1) and row == 0:
                    crop_image = current_image[margin_y//2:y-margin_y//2, 0:x-margin_x//2]
                    full_image1[y_start+margin_y//2:y_end+margin_y//2, x_start:x_end + margin_x//2] = crop_image
                elif col==last_y-1 and row==0:
                    crop_image = current_image[margin_y//2:y, 0:x - margin_x//2]
                    full_image1[y_start+margin_y//2:y_end+margin_y, x_start:x_end+margin_x//2] = crop_image
                elif (col>0 and col<last_y-1) and (row>0 and row < last_x-1):
                    crop_image = current_image[margin_y//2:y-margin_y//2, margin_x//2:x - margin_x//2]
                    full_image1[y_start+margin_y//2:y_end+margin_y//2, x_start+margin_x//2:x_end+margin_x//2] = crop_image
                elif col==last_y-1 and row>0 and (row < last_x-1):
                    crop_image = current_image[margin_y//2:y, margin_x//2:x - margin_x//2]
                    full_image1[y_start+margin_y//2:y_end+margin_y, x_start+margin_x//2:x_end+margin_x//2] = crop_image
                elif (col>0 and col<last_y-1) and row==last_x-1:
                    crop_image = current_image[margin_y//2:y-margin_y//2, margin_x//2:x]
                    full_image1[y_start+margin_y//2:y_end+margin_y//2, x_start+margin_x//2:x_end+margin_x] = crop_image
                elif col==last_y-1 and row==last_x-1:
                    crop_image = current_image[margin_y // 2:y, margin_x // 2:x]
                    full_image1[y_start + margin_y // 2:y_end + margin_y,x_start + margin_x // 2:x_end + margin_x] = crop_image
        for row in range(last_x):
            for col in range(last_y):

                suffix =  f'_x{row+1:0>3}_y{col+1 :0>3}.tiff'
                current_image = imread(str(location+'/'+file_name+suffix))
                x_start = int((row)) * (x - margin_x)
                y_start = int((col)) * (y - margin_y)
                x_end = (row+1) * (x - margin_x)
                y_end = (col+1) * (y - margin_y)
                if row == last_x-1 and col != last_y-1:
                    crop_image = current_image[0:y - margin_y, 0:x]
                    full_image2[y_start:y_end, x_start:x_end + margin_x] = crop_image
                elif col == last_y-1 and row != last_x-1:
                    crop_image = current_image[0:y, 0:x - margin_x]
                    full_image2[y_start:y_end + margin_y, x_start:x_end] = crop_image
                elif col == last_y-1 and row == last_x-1:
                    crop_image = current_image[0:y, 0:x]
                    full_image2[y_start:y_end + margin_y, x_start:x_end + margin_x] = crop_image
                else:
                    crop_image = current_image[0:y - margin_y, 0:x - margin_x]
                    full_image2[y_start:y_end, x_start:x_end] = crop_image

        ag1=full_image2
        ag2=full_image1




        avg_cv1 = cv2.addWeighted(ag2, 0.5, ag1, 0.5, 0)
        avg_cv2 = cv2.addWeighted(ag2, 0.5, ag1, 0.5, 0)
        for g in range(10):#nie potrzeba robić w pętli
            avg_cv4 = cv2.addWeighted(avg_cv2, 0.5, avg_cv1, 0.5, 0)
            avg_cv3 = cv2.addWeighted(avg_cv2, 0.5, avg_cv1, 0.5, 0)

            avg_cv2 = avg_cv3
            avg_cv1 = avg_cv4


        bilateral_filtered_image = cv2.bilateralFilter(avg_cv1, 7, 150, 5)


        return bilateral_filtered_image


