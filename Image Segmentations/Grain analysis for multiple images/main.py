import cv2
import glob
import grain_anaysis as ga



# main function
def main():
    
    # pixel to micron conversion factor.
    pixel_to_micron = 0.5 # 1 pixel = 0.5 micron.
    
    
    # prop_list is a list of properties that we want to measure for each grain.
    propList = ['Area',
                'equivalent_diameter',
                'orientation',
                'MajorAxisLength',
                'MinorAxisLength',
                'Perimeter','MinIntensity',
                'MeanIntensity',
                'MaxIntensity']
    
    # open a csv file to save the results.
    out_file = open('./grains_properties.csv', 'w')
    out_file.write(('FileName' +  "," + 'Grain # ' +  ',' + ','.join(propList) + '\n')) # write the header of the csv file.
    
    
    # loop over all the images.
    for img in glob.glob("./images/*.jpg"):
      
        # print the name of the file.
        print(img)
        
        # read the image.
        img1 = cv2.imread(img)
        
        # convert the image to grayscale.
        
        img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        

        # call the grain segmentation function.
        regions = ga.grain_segmentation(img)

        grain_num = 1 # initialize the grain number.
        
        
        
        # loop over each grain.
        for region_props in regions:
            
            # write the file name and grain number to the csv file.
            out_file.write(img + ',' + str(grain_num))
            
            # loop over each property of the grain.
            for i,prop in enumerate(propList):
                
                # write the property of the grain to the csv file.
                if(prop == 'Area'): 
                    to_print = region_props[prop]*pixel_to_micron**2
                elif(prop == 'orientation'):
                    to_print = region_props[prop]*57.2958 # convert to degrees from radians.
                elif(prop.find('Intensity') < 0):
                    to_print = region_props[prop]*pixel_to_micron # convert to micron.
                else:
                    to_print = region_props[prop]
                out_file.write(',' + str(to_print)) # write the property of the grain to the csv file.
            out_file.write('\n') # write a new line.
            grain_num += 1 # increment the grain number.
    out_file.close() # close the csv file.
    

if __name__ == '__main__':
    main()
    