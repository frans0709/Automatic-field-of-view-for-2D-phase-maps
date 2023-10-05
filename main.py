import Preprocessing
import tkinter as tk
from tkinter import filedialog, messagebox
from skimage.io import imsave, imread
import datetime
import numpy as np
import Files_module
import tifffile
class GUI:
    def __init__(self, master):
        self.master = master
        self.master.geometry("950x600")
        self.master.title("Stitching 2D")

        # create frames
        self.frame1 = tk.Frame(self.master, width=600, height=600)
        self.frame1.grid(row=0, column=0)
        self.frame2 = tk.Frame(self.master, width=350, height=350)
        self.frame2.grid(row=0, column=1, sticky="n")
        self.frame3=tk.Frame(self.master,width=350, height=250)
        self.frame3.grid(row=0, column=1, sticky="s")

        tk.Frame(self.master, bg='black', width=6, height=600).place(x=594,y=0)
        tk.Frame(self.master, bg='black', width=6, height=600).place(x=0, y=0)
        tk.Frame(self.master, bg='black', width=6, height=600).place(x=944, y=0)
        tk.Frame(self.master, bg='black', width=950, height=6).place(x=0, y=0)
        tk.Frame(self.master, bg='black', width=350, height=6).place(x=600, y=347)

        tk.Frame(self.master, bg='black', width=950, height=6).place(x=0, y=594)
        #tk.Frame(self.master, bg='black', width=600, height=4).place(x=0, y=476)
        # create widgets in frame1
        self.label_title = tk.Label(self.frame1, text="Preparing images:", font=("Cambria", 16))
        self.label_title.place(relx=0.5, rely=0.05, anchor="center")

        self.label_file_name = tk.Label(self.frame1, text="File name:")
        self.label_file_name.place(x=40,y=70)
        self.entry_file_name = tk.Entry(self.frame1, width=20)
        self.entry_file_name.place(x=200,y=70)
        self.entry_file_name.insert(0,"holo")

        self.folder_path = tk.StringVar()
        self.label_main_path = tk.Label(self.frame1, text="Localization:")
        self.label_main_path.place(x=40, y=100)
        self.entry_main_path = tk.Entry(self.frame1,textvariable=self.folder_path, width=40)
        self.entry_main_path.place(x=200, y=100)
        browse_button = tk.Button(self.frame1, text='Chose folder', command=self.browse_folder)
        browse_button.place(x=460,y=95)

        self.directory_path = tk.StringVar()
        self.label_directory = tk.Label(self.frame1, text="Destination folder:")
        self.label_directory.place(x=40, y=130)
        self.entry_directory = tk.Entry(self.frame1,textvariable=self.directory_path, width=40)
        self.entry_directory.place(x=200, y=130)
        browse_button_directory = tk.Button(self.frame1, text='Chose folder', command=self.browse_directory)
        browse_button_directory.place(x=460, y=125)

        self.label_Y = tk.Label(self.frame1, text="Columns:")
        self.label_first_y = tk.Label(self.frame1, text="first_x")
        self.label_last_y = tk.Label(self.frame1, text="last_x")
        self.label_first_y.place(x=160, y=160)
        self.label_last_y.place(x=280, y=160)
        self.label_Y.place(x=40,y=160)
        self.entry_first_y = tk.Entry(self.frame1, width=5)
        self.entry_first_y.insert(0, 1)
        self.entry_first_y.place(x=210,y=160)
        self.entry_last_y = tk.Entry(self.frame1, width=5)
        self.entry_last_y.insert(0, 14)
        self.entry_last_y.place(x=330, y=160)

        self.label_X = tk.Label(self.frame1, text="Rows:")
        self.label_first_x = tk.Label(self.frame1, text="first_y")
        self.label_last_x = tk.Label(self.frame1, text="last_y")
        self.label_first_x.place(x=160, y=190)
        self.label_last_x.place(x=280, y=190)
        self.label_X.place(x=40, y=190)
        self.entry_first_x = tk.Entry(self.frame1, width=5)
        self.entry_first_x.insert(0, 1)
        self.entry_first_x.place(x=210, y=190)
        self.entry_last_x = tk.Entry(self.frame1, width=5)
        self.entry_last_x.insert(0, 16)
        self.entry_last_x.place(x=330, y=190)

        self.label_FFT = tk.Label(self.frame1,text='Microscope settings:',font=("Cambria",12))
        self.label_FFT.place(x=300,y=230, anchor="center")

        self.label_magnification = tk.Label(self.frame1, text="Magnification:")
        self.label_magnification.place(x=40, y=260)
        self.entry_magnification = tk.Entry(self.frame1, width=10)
        self.entry_magnification.place(x=200, y=260)
        self.entry_magnification.insert(0, 16.667)

        self.label_wavelenght = tk.Label(self.frame1, text="Wavelenght (nm):")
        self.label_wavelenght.place(x=40, y=290)
        self.entry_wavelenght = tk.Entry(self.frame1, width=10)
        self.entry_wavelenght.place(x=200, y=290)
        self.entry_wavelenght.insert(0, 0.633)

        self.label_NA = tk.Label(self.frame1, text="Numeric aperture:")
        self.label_NA.place(x=340, y=260)
        self.entry_NA = tk.Entry(self.frame1, width=10)
        self.entry_NA.place(x=500, y=260)
        self.entry_NA.insert(0, 0.45)

        self.label_px_size = tk.Label(self.frame1, text="Pixel size (um):")
        self.label_px_size.place(x=340, y=290)
        self.entry_px_size = tk.Entry(self.frame1, width=10)
        self.entry_px_size.place(x=500, y=290)
        self.entry_px_size.insert(0, 3.45)

        self.param_button = tk.Button(self.frame1, text='Get phase',font=('Calibri',10,'bold'), command=self.run_parameters)
        self.param_button.place(x=300, y=330, anchor="center")

        self.label_FFT = tk.Label(self.frame1, text='Preprocessing:', font=("Cambria", 12))
        self.label_FFT.place(x=300,y=365, anchor="center")

        self.label_file_name_pp = tk.Label(self.frame1, text="File name:")
        self.label_file_name_pp.place(x=40,y=400)
        self.entry_file_name_pp = tk.Entry(self.frame1, width=20)
        self.entry_file_name_pp.place(x=200,y=400)
        self.entry_file_name_pp.insert(0,"phase_")
        self.folder_path_pp = tk.StringVar()

        self.label_main_path_pp = tk.Label(self.frame1, text="Localization holo:")
        self.label_main_path_pp.place(x=40, y=430)
        self.entry_main_path_pp = tk.Entry(self.frame1, textvariable=self.folder_path_pp, width=30)
        self.entry_main_path_pp.place(x=200, y=430)
        browse_button_pp = tk.Button(self.frame1, text='Chose folder', command=self.browse_folder_pp)
        browse_button_pp.place(x=400, y=425)

        self.directory_path_pp = tk.StringVar()
        self.label_directory_pp = tk.Label(self.frame1, text="Destination folder:")
        self.label_directory_pp.place(x=40, y=460)
        self.entry_directory_pp = tk.Entry(self.frame1, textvariable=self.directory_path_pp, width=30)
        self.entry_directory_pp.place(x=200, y=460)
        browse_button_directory_pp = tk.Button(self.frame1, text='Chose folder', command=self.browse_directory_pp)
        browse_button_directory_pp.place(x=400, y=455)

        self.param_button_pp = tk.Button(self.frame1, text='Start preprocessing',font=('Calibri',10,'bold'), command=self.run_parameters_pp)
        self.param_button_pp.place(x=400, y=530, anchor="center")



        self.label_overlap = tk.Label(self.frame1, text="Overlap:")
        self.label_overlap.place(x=40, y=490)
        self.entry_overlap = tk.Entry(self.frame1, width=10)
        self.entry_overlap.place(x=200, y=490)
        self.entry_overlap.insert(0, 0.3)

        self.label_LI = tk.Label(self.frame1, text="Balance factor Li:")
        self.label_LI.place(x=40, y=520)
        self.entry_LI = tk.Entry(self.frame1, width=10)
        self.entry_LI.place(x=200, y=520)
        self.entry_LI.insert(0, 0.2)

        self.label_I = tk.Label(self.frame1, text="Iterations:")
        self.label_I.place(x=40, y=550)
        self.entry_I = tk.Entry(self.frame1, width=10)
        self.entry_I.place(x=200, y=550)
        self.entry_I.insert(0, 50)

        #====================================================================================

        # utworzenie etykiety
        self.label_AG = tk.Label(self.frame3, text="Stitching:",font=("Cambria",14))
        self.label_AG.place(x=175,y=15,anchor="center")

        self.label_AG_file = tk.Label(self.frame3, text="File name:")
        self.label_AG_file.place(x=10, y=45)
        self.entry_AG_file = tk.Entry(self.frame3, width=20)
        self.entry_AG_file.place(x=95, y=45)
        self.entry_AG_file.insert(0,'After_preproc')


        self.stitching_path = tk.StringVar()
        self.label_stitching_directory= tk.Label(self.frame3, text="Path:")
        self.label_stitching_directory.place(x=10, y=75)
        self.entry_stitching_directory = tk.Entry(self.frame3, textvariable=self.stitching_path, width=25)
        self.entry_stitching_directory.place(x=95, y=75)
        browse_button = tk.Button(self.frame3, text='Chose folder', command=self.browse_stitching_directory)
        browse_button.place(x=260, y=70)

        # utworzenie etykiety
        self.label_AG = tk.Label(self.frame3, text="Stitching method:")
        self.label_AG.place(x=175, y=130, anchor="center")

        # utworzenie pola z rozwijaną listą opcji
        self.option_var = tk.StringVar(value='One-side cropping')  # ustawienie domyślnej wartości
        self.option_menu = tk.OptionMenu(self.frame3, self.option_var, 'One-side cropping', 'Two-side cropping',
                                         'Linear blending', 'Linear blending v2', 'Linear blending for full images')
        self.option_menu.place(x=175, y=160, anchor="center")
        self.param_button_AG = tk.Button(self.frame3, text='Stitch images!',font=('Calibri',10,'bold'), command=self.chose_arrange_method)
        self.param_button_AG.place(x=175, y=200, anchor="center")

        #===============================================================================================================

        self.label_H5 = tk.Label(self.frame2, text="Data from .H5:", font=("Cambria", 12))
        self.label_H5.place(x=150, y=30, anchor="center")
        self.label_H5_path = tk.Label(self.frame2, text="File name:")
        self.label_H5_path.place(x=15, y=60)
        self.entry_H5_path = tk.Entry(self.frame2, width=20)
        self.entry_H5_path.place(x=90, y=60)

        self.folder_path_2d = tk.StringVar()
        self.label_H5_folder = tk.Label(self.frame2, text="Localization:")
        self.label_H5_folder.place(x=15, y=90)
        self.entry_H5_folder = tk.Entry(self.frame2, width=25)
        self.entry_H5_folder.place(x=90, y=90)
        browse_button = tk.Button(self.frame2, text='Chose folder', command=self.browse_folder_H5)
        browse_button.place(x=260, y=85)
        self.param_button = tk.Button(self.frame2, text='Start', font=('Calibri', 10, 'bold'),
                                      command=self.run_parameters_H5)
        self.param_button.place(x=280, y=125, anchor="center")
        #===============================================================================================================

        self.label_MAT = tk.Label(self.frame2, text="Data from .MAT:", font=("Cambria", 12))
        self.label_MAT.place(x=150, y=130, anchor="center")
        self.label_MAT_path = tk.Label(self.frame2, text="File name:")
        self.label_MAT_path.place(x=15, y=160)
        self.entry_MAT_path = tk.Entry(self.frame2, width=20)
        self.entry_MAT_path.place(x=90, y=160)

        self.folder_path_2d = tk.StringVar()
        self.label_MAT_folder = tk.Label(self.frame2, text="Localization:")
        self.label_MAT_folder.place(x=15, y=190)
        self.entry_MAT_folder = tk.Entry(self.frame2, width=25)
        self.entry_MAT_folder.place(x=90, y=190)
        browse_button = tk.Button(self.frame2, text='Chose folder', command=self.browse_folder_H5)
        browse_button.place(x=260, y=185)
        self.param_button = tk.Button(self.frame2, text='Start', font=('Calibri', 10, 'bold'),
                                      command=self.run_parameters_mat)
        self.param_button.place(x=280, y=225, anchor="center")

        #===============================================================================================================
        self.label_MAT = tk.Label(self.frame2, text="Multilayer files:", font=("Cambria", 12))
        self.label_MAT.place(x=150, y=240, anchor="center")

        self.label_2d_file_name = tk.Label(self.frame2, text="File name:")
        self.label_2d_file_name.place(x=15, y=260)
        self.entry_file_name_2d = tk.Entry(self.frame2, width=20)
        self.entry_file_name_2d.place(x=90, y=260)

        self.folder_path_2d = tk.StringVar()
        self.label_main_path_2d = tk.Label(self.frame2, text="Localization:")
        self.label_main_path_2d.place(x=15, y=290)
        self.entry_main_path_2d = tk.Entry(self.frame2, textvariable=self.folder_path_2d, width=25)
        self.entry_main_path_2d.place(x=90, y=290)
        browse_button = tk.Button(self.frame2, text='Chose folder', command=self.browse_folder_2d)
        browse_button.place(x=260, y=285)

        self.label_layer_2d=tk.Label(self.frame2, text="Layer no.")
        self.label_layer_2d.place(x=15,y=320)
        self.entry_layer_2d=tk.Entry(self.frame2,width=5)
        self.entry_layer_2d.place(x=90,y=320)
        self.entry_layer_2d.insert(0,'1')
        self.param_button = tk.Button(self.frame2, text='Start', font=('Calibri', 10, 'bold'),
                                      command=self.run_parameters_2d)
        self.param_button.place(x=280, y=330, anchor="center")
        #===============================================================================================================
    def browse_folder(self):
        folder_path = filedialog.askdirectory()
        self.entry_main_path.delete(0, tk.END)
        self.entry_main_path.insert(0, folder_path)
    def browse_folder_2d(self):
        folder_path = filedialog.askdirectory()
        self.entry_main_path_2d.delete(0, tk.END)
        self.entry_main_path_2d.insert(0, folder_path)
    def browse_folder_H5(self):
        folder_path = filedialog.askdirectory()
        self.entry_H5_folder.delete(0, tk.END)
        self.entry_H5_folder.insert(0, folder_path)

    def browse_directory(self):
        folder_path2 = filedialog.askdirectory()
        self.entry_directory.delete(0, tk.END)
        self.entry_directory.insert(0, folder_path2)
        #self.entry_main_path_pp.delete(0, tk.END)
        #self.entry_main_path_pp.insert(0, folder_path2)
    def browse_folder_pp(self):
        folder_path3 = filedialog.askdirectory()
        self.entry_main_path_pp.delete(0, tk.END)
        self.entry_main_path_pp.insert(0, folder_path3)
    def browse_directory_pp(self):
        folder_path2 = filedialog.askdirectory()
        self.entry_directory_pp.delete(0, tk.END)
        self.entry_directory_pp.insert(0, folder_path2)
        self.entry_stitching_directory.delete(0, tk.END)
        self.entry_stitching_directory.insert(0, folder_path2)
    def browse_stitching_directory(self):
        folder_path = filedialog.askdirectory()
        self.entry_stitching_directory.delete(0, tk.END)
        self.entry_stitching_directory.insert(0, folder_path)

    def print_selection(self):
        print("Wybrano:", self.combo_var.get())
    def run_parameters(self):

        first_y = int(self.entry_first_x.get())
        last_y = int(self.entry_last_x.get())
        first_x = int(self.entry_first_y.get())
        last_x = int(self.entry_last_y.get())
        main_path = self.entry_main_path.get()
        direction = self.entry_directory.get()
        file_name = self.entry_file_name.get()
        NA = float(self.entry_NA.get())
        magnification = float(self.entry_magnification.get())
        wavelenght = float(self.entry_wavelenght.get())
        px_size = float(self.entry_px_size.get())


        paths, params_FFT= Preprocessing.parameters_fft(first_x, last_x, first_y,last_y,main_path,
                                                                    file_name,NA, magnification, wavelenght,px_size)


        first_x = params_FFT['first_x']
        first_y = params_FFT["first_y"]
        last_x = params_FFT["last_x"]
        last_y = params_FFT["last_y"]


        reverse_counter = last_y - 1

        path_iteration = 0
        for row in range(last_x):
            for col in range(last_y):
                postfix = f'_x{row + first_x:0>3}_y{col + first_y:0>3}.tiff'
                # full_name=name+suffix
                full_path_from = paths[path_iteration]
                path_iteration += 1

                image = imread(full_path_from)#[0]
                start = Preprocessing.Fourier_Transform(height=image.shape[0], width=image.shape[1],
                                                        NA=params_FFT["NA"],
                                                        magnification=params_FFT["magnification"],
                                                        wavelength=params_FFT["wavelenght"],
                                                        px_size=params_FFT["px_size"])
                start.holo = image
                start.Tukey_Window()
                im0 = start.apodization_filter
                spectrum_test = start.Get_Spectrum()
                start.Detect_Information_Peak()

                start.Calculate_Information_Area()
                start.ph_ref()
                start.Tukey_Window2()
                im3 = start.IFFT2()
                crop_img = im3[1:im3.shape[0] - 1, 1:im3.shape[1] - 1]
                tifffile.imsave(str(direction+'/'+'phase'+postfix), crop_img)
                """scaled3 = (255 * (crop_img - np.min(crop_img)) / (np.max(crop_img) - np.min(crop_img))).astype(np.uint8)
                rounded3 = np.around(scaled3).astype(np.uint8)
                imsave(str(direction +'/'+ 'phase' + postfix), rounded3)"""




    def run_parameters_pp(self):

        first_y = int(self.entry_first_x.get())
        last_y = int(self.entry_last_x.get())
        first_x = int(self.entry_first_y.get())
        last_x = int(self.entry_last_y.get())
        main_path = self.entry_main_path_pp.get()
        direction = self.entry_directory_pp.get()
        file_name = self.entry_file_name_pp.get()
        overlap = float(self.entry_overlap.get())
        LI = float(self.entry_LI.get())
        I = int(self.entry_I.get())

        paths, params = Preprocessing.parameters_pp(overlap, first_x, last_x, first_y, last_y, main_path, direction,
                                                     file_name,LI,I)



        images = Preprocessing.ImageCollection(paths, params)

        reverse_counter=last_y-1
        for row in range(last_x):
            for col in range(last_y):
                suffix = f'_x{row + first_x:0>3}_y{col + first_y:0>3}.tiff'
                images.table_with_im[reverse_counter - col][row] = imread(
                    main_path+"/"+file_name + suffix)

        avg_abr = images.return_average_aberrations()
        images.remove_precalc_average_aberrations(avg_abr)
        images.gradient_slopes()
        images.baseline_correction()
        messagebox.showinfo("Informacja", "Czynność została zakończona!")

    def chose_arrange_method(self):
        file_name=str(self.entry_AG_file.get())
        location=str(self.entry_stitching_directory.get())
        method=self.option_var.get()
        overlap = float(self.entry_overlap.get())
        first_y = int(self.entry_first_x.get())
        last_y = int(self.entry_last_x.get())
        first_x = int(self.entry_first_y.get())
        last_x = int(self.entry_last_y.get())
        file_name,location,method=Preprocessing.AG_parameters(file_name, location, method)

        if method==1:
            image=Preprocessing.ImageCollection.arrange_grid(self,file_name,location,overlap,first_y,first_x,last_y,last_x)
            now = datetime.datetime.now()
            filename = now.strftime("_%d-%m-%Y-%H-%M")
            name5 = location+'/' + '_Stitched_img_AG1_'+ filename +".tiff"
            imsave(str(name5), image)
            messagebox.showinfo("Informacja", "One-side cropping is ended")
        elif method==2:
            image = Preprocessing.ImageCollection.arrange_grid2(self, file_name, location, overlap, first_y, first_x,last_y, last_x)
            now = datetime.datetime.now()
            filename = now.strftime("_%d-%m-%Y-%H-%M")
            name5 = location + '/' + '_Stitched_img_AG2_' + filename + ".tiff"
            imsave(str(name5), image)
            messagebox.showinfo("Informacja", "Two-side cropping is ended")
        elif method==3:
            image = Preprocessing.ImageCollection.arrange_grid3(self, file_name, location, overlap, first_y, first_x,
                                                                last_y, last_x)
            now = datetime.datetime.now()
            filename = now.strftime("_%d-%m-%Y-%H-%M")
            name5 = location + '/' + '_Stitched_img_AG3_' + filename + ".tiff"
            imsave(str(name5), image)
            messagebox.showinfo("Informacja", "Linear blending is ended")
        elif method==4:
            image = Preprocessing.ImageCollection.arrange_grid4(self, file_name, location, overlap, first_y, first_x,
                                                                last_y, last_x)
            now = datetime.datetime.now()
            filename = now.strftime("_%d-%m-%Y-%H-%M")
            name5 = location + '/' + '_Stitched_img_AG4_' + filename + ".tiff"
            imsave(str(name5), image)
            messagebox.showinfo("Informacja", "Linear blending v2 is ended")
        elif method==5:
            image = Preprocessing.ImageCollection.arrange_grid5(self, file_name, location, overlap, first_y, first_x,
                                                                last_y, last_x)
            now = datetime.datetime.now()
            filename = now.strftime("_%d-%m-%Y-%H-%M")
            name5 = location + '/' + '_Stitched_img_AG5_' + filename + ".tiff"
            imsave(str(name5), image)
            messagebox.showinfo("Informacja", "Linear blending for full images is ended")

    def run_parameters_2d(self):

        first_y = int(self.entry_first_x.get())
        last_y = int(self.entry_last_x.get())
        first_x = int(self.entry_first_y.get())
        last_x = int(self.entry_last_y.get())
        main_path = self.entry_main_path_2d.get()
        file_name = self.entry_file_name_2d.get()
        layer=int(self.entry_layer_2d.get())

        Files_module.process_images_2D(first_x,first_y,last_x,last_y,file_name,main_path,layer)
        messagebox.showinfo("Info", "Done!")
    def run_parameters_mat(self):


        path=self.entry_MAT_folder.get()
        file_name=self.entry_MAT_path.get()

        Files_module.save_mat_as_tiff(file_name,path)
        messagebox.showinfo("Info", "Done!")
    def run_parameters_H5(self):


        path=self.entry_H5_folder.get()
        file_name=self.entry_H5_path.get()

        Files_module.save_h5_as_tiff(file_name,path)
        messagebox.showinfo("Info", "Done!")



# Pobranie wartości z pól tekstowych i menu

def main():
    root = tk.Tk()
    gui = GUI(root)
    root.mainloop()
if __name__ == '__main__':
    main()





