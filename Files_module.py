from skimage.io import imread, imsave
import h5py
import scipy.io

#zapis przekrojów 2d .tiff z wielowarstowych plików .tiff
def process_images_2D(first_x: int, first_y: int, last_x: int, last_y: int, name: str, path_from: str, layer: int = 0):
    for row in range(last_x):
        for col in range(last_y):
            postfix = f'_x{row + first_x:0>3}_y{col + first_y:0>3}.tiff'
            full_name = name + postfix
            full_path_from = path_from +'/'+ full_name

            img = imread(full_path_from)[layer-1]
            imsave(str(path_from + '/2D_'+full_name), img)

#zapis przekrojów z plików .H5
def save_h5_as_tiff(file_name,path):
    postfix='.h5'
    with h5py.File(path+'/'+file_name+postfix, 'r') as file:
        center_hologram = file['/measurement/Center_hologram'][()]
        galvo_x = file['/measurement'].attrs['galvo_x']
        galvo_y = file['/measurement'].attrs['galvo_y']
        file.close()
    new_name = file_name+f"x{int(galvo_x):03}_y{int(galvo_y):03}.tiff"
    imsave(str(path+'/'+ new_name), center_hologram)

#zapis przekrojów z plików .MAT
def save_mat_as_tiff(file_name, path):
    postfix='.mat'
    data = scipy.io.loadmat(path+'/'+file_name+postfix)
    structure = data['ph']

    # Zapisywanie struktury jako pliku .tiff
    imsave(str(path+'/2D_'+ file_name+'.tiff'), structure)

