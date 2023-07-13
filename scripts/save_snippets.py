# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 08:16:10 2018

@author: smith
"""
##### no longer used, i think
    ## Label centers is an array of tuples
    #label_centers = measurements.center_of_mass(bin_image, labels=label_im, index=labels_tmp)
    ##centers = measurements.center_of_mass(label_im, labels=label_clean, index=None)
    ## x-axis going right to left in the image is the second element of the centers tuple
    ## y-axis going up and down is the first element in the centers tuple
    #x = []
    #y = []
    #label_centers_array = []
    #for index in range(0, len(label_centers)) :
    #        x.append(int(label_centers[index][1]))
    #        y.append(int(label_centers[index][0]))
    #        # switch the coordinates to (x,y)=(right-left, up-down) to pass back
    #        label_centers_array.append([int(label_centers[index][1]), int(label_centers[index][0])])
    #print("old label centers: ",label_centers_array )
#####################
    




           #print(df_images.tail(1))
           UseOldObjectHandling = False
           if UseOldObjectHandling :   # write out duplicate data for seq_len = 2 because matching with label file needs both

               save_morph_in_seq.append(img_morph)
               objects_centers.append(centers)

           # Save the dataframe unless this is the second in a sequence of 2. Differences 0-1 and 1-0 are identical.
           #if not ( (seq_index+1) == seq_len and seq_len == 2 ) :
               # save header record if header and detail records are being used
               #df_images = df_images.append({'File':df_save_sequence.loc[seq_index, 'File'], 'Dir':df_save_sequence.loc[seq_index,'Dir'], 'Datetime':df_save_sequence.loc[seq_index,'Datetime'], 'Camera':df_save_sequence.loc[seq_index,'Camera'], 'SeqNum':df_save_sequence.loc[seq_index,'SeqNum'], 'SeqLen':df_save_sequence.loc[seq_index,'SeqLen'], 'SeqNumDiff':df_save_sequence.loc[seq_index,'SeqNumDiff'] , 'Night':0, 'Mean':df_save_sequence.loc[seq_index,'Mean'], 'Std':df_save_sequence.loc[seq_index,'Std'], 'TopCrop': top_crop, 'BottomCrop':bottom_crop, 'Carcass X':deer_x, 'Carcass Y':deer_y, 'Carcass Dist':deer_dist, 'Carcass Size':deer_size, 'Obscuring Plants':deer_plants, 'NumObj':df_save_sequence.loc[seq_index,'NumObj'], 'DistRank':0}, ignore_index=True)
               #print("sorting for seq_index, nbr", seq_index, nbr)


               # distance is needed to determine the closest object
               # angle can be calculated from the coordinates in the machine learning program
               objects_dist = []
               objects_angle = []
               # objects found
               for obj_index in range(0, nbr) :
                   obj_coord = centers[obj_index]             # this worked with centers from above
                   obj_x = obj_coord[0]
                   obj_y = obj_coord[1]
                   obj_size = sizes[obj_index]
                   #print("distance coordinates: ", obj_coord[0], obj_coord[1])
                   #obj_dist = int(math.sqrt((obj_x - deer_x)**2+(obj_y - deer_y)**2))
                   obj_dist = int(math.sqrt((obj_x - deer_x)**2+(obj_y - deer_y)**2))

                   #if obj_coord[0]== deer_x :
                   if obj_coord[0]== deer_x :
                       obj_coord[0] = obj_coord[0] + 1   # fudge to avoid divide by 0
                      
                   obj_angle = int(math.degrees(math.atan(-(obj_coord[1] - deer_y)/(obj_coord[0] - deer_x))))
      
                    # looks like these arrays don't work
                   objects_dist.append(obj_dist)
                   objects_angle.append(obj_angle)
                   #print("obj_angle: ", obj_angle, "objects_angle: ", objects_angle)
                   ######
                   df_object = df_object.append({'Size':sizes[obj_index], 'X':obj_coord[0], 'Y':obj_coord[1], 'Dist':obj_dist, 'Angle':obj_angle}, ignore_index=True)

               #print("Seq_index: ", seq_index, "  ImagePair: " , str(df_save_sequence.loc[seq_index,'ImgNum'])+'|'+str(df_save_sequence.loc[seq_index,'ImgNum']+1))
               #print(df_object)
               size_mean, size_std, size_skew = 0, 0, 0
               dist_mean, dist_std, dist_skew = 0, 0, 0
               angle_mean, angle_std, angle_skew = 0, 0, 0
               if nbr > 0 :
                   size_mean, size_std, size_skew = df_object.Size.mean(), df_object.Size.std(), df_object.Size.skew()
                   dist_mean, dist_std, dist_skew =  df_object.Dist.mean(), df_object.Dist.std(), df_object.Dist.skew()
                   angle_mean, angle_std, angle_skew = df_object.Angle.mean(), df_object.Angle.std(), df_object.Angle.skew()
           
                   
               #print("angle_mean: ", angle_mean,"angle_std: ", angle_std, "angle_skew: ", angle_skew )

                   #print("distance, angle: ", objects_dist[obj_index], objects_angle[obj_index])
                   # add a detail record to df_images for each object in the image
               # Use this for closest object data in single record. This case is no objects found.
               if nbr == 0 :
                   #df_images = df_images.append({'File':df_save_sequence.loc[seq_index, 'File'], 'Dir':df_save_sequence.loc[seq_index,'Dir'], 'Datetime':df_save_sequence.loc[seq_index,'Datetime'], 'Camera':df_save_sequence.loc[seq_index,'Camera'], 'SeqNum':df_save_sequence.loc[seq_index,'SeqNum'], 'SeqLen':df_save_sequence.loc[seq_index,'SeqLen'], 'SeqNumDiff':df_save_sequence.loc[seq_index,'SeqNumDiff'] , 'Night':0, 'Mean':df_save_sequence.loc[seq_index,'Mean'], 'Std':df_save_sequence.loc[seq_index,'Std'], 'TopCrop': top_crop, 'BottomCrop':bottom_crop, 'Carcass X':deer_x, 'Carcass Y':deer_y, 'Carcass Dist':deer_dist, 'Carcass Size':deer_size, 'Obscuring Plants':deer_plants, 'NumObj':df_save_sequence.loc[seq_index,'NumObj'], 'DistRank':0}, ignore_index=True)

                   df_images = df_images.append({'Dir':df_save_sequence.loc[seq_index,'Dir'], 
                                                 'Camera':df_save_sequence.loc[seq_index,'Camera'], 
                                                 'UpperX':int(upperleft_x), 
                                                 'UpperY':int(upperleft_y), 
                                                 'LowerX':int(lowright_x), 
                                                 'LowerY':int(lowright_y), 
                                                 'ImagePair':get_image_pair(df_save_sequence.loc[seq_index,'ImgNum'], seq_index, seq_num, seq_len), 
                                                 'Datetime':df_save_sequence.loc[seq_index,'Datetime'], 
                                                 'SeqNum':df_save_sequence.loc[seq_index,'SeqNum'], 
                                                 'SeqLen':df_save_sequence.loc[seq_index,'SeqLen'], 
                                                 'Night':0, 
                                                 'DiffMean':round(df_save_sequence.loc[seq_index,'DiffMean'], 0), 
                                                 'DiffStd':round(df_save_sequence.loc[seq_index,'DiffStd'], 0), 
                                                 'DiffMedian':round(df_save_sequence.loc[seq_index,'DiffMedian'], 0), 
                                                 'NumObj':df_save_sequence.loc[seq_index,'NumObj'], 
                                                 'TimeDiff1': get_timediff(seq_time_stamp[-2], df_save_sequence.loc[seq_index,'Datetime']),
                                                 'TimeDiff2': get_timediff(seq_time_stamp[-3], df_save_sequence.loc[seq_index,'Datetime']), 
                                                 'TimeDiff3': get_timediff(seq_time_stamp[-4], df_save_sequence.loc[seq_index,'Datetime']), 
                                                 }, ignore_index=True)



               else :
                   # Closest object no longer used. Replaced by aggregate size, dist, angle.
                   df_object['DistRank'] = df_object['Dist'].rank(ascending=True)
                   index_closest = 0
                   if (df_object.DistRank== 1).any() :
                       index_closest = df_object[df_object.DistRank == 1].index.tolist()[0]
                   elif (df_object.DistRank == 1.5).any() :
                        # This is arbitrary and not a good way to handle it.
                       index_closest = df_object[df_object.DistRank == 1.5].index.tolist()[0]
                   else :   # should never get here due to the nbr=0 check above
                       print("!!!!!!! ERROR: Found image with no closest object", df_save_sequence.loc[seq_index, 'File'])
                       #df_images = df_images.append({'File':df_save_sequence.loc[seq_index, 'File'], 
                       #                              'Dir':df_save_sequence.loc[seq_index,'Dir'], 
                       #                              'Datetime':df_save_sequence.loc[seq_index,'Datetime'], 
                       #                              'Camera':df_save_sequence.loc[seq_index,'Camera'], 
                       #                              'SeqNum':df_save_sequence.loc[seq_index,'SeqNum'], 
                       #                              'SeqLen':df_save_sequence.loc[seq_index,'SeqLen'], 
                       #                              'SeqNumDiff':df_save_sequence.loc[seq_index,'SeqNumDiff'] , 
                       #                              'Night':0, 
                       #                              'Mean':df_save_sequence.loc[seq_index,'Mean'], 
                       #                              'Std':df_save_sequence.loc[seq_index,'Std'], 'TopCrop': top_crop, 'BottomCrop':bottom_crop, 'Carcass X':deer_x, 'Carcass Y':deer_y, 'Carcass Dist':deer_dist, 'Carcass Size':deer_size, 'Obscuring Plants':deer_plants, 'NumObj':df_save_sequence.loc[seq_index,'NumObj'], 'DistRank':0}, ignore_index=True)

                   
                   # Add single record with closest object data
                   df_images = df_images.append({'Dir':df_save_sequence.loc[seq_index,'Dir'], 
                                                 'Camera':df_save_sequence.loc[seq_index,'Camera'], 
                                                 'ROI': [(int(upperleft_x),int(upperleft_y)), (int(lowright_x),int(lowright_y))],
                                                 #'UpperX':int(upperleft_x), 
                                                 #'UpperY':int(upperleft_y), 
                                                 #'LowerX':int(lowright_x), 
                                                 #'LowerY':int(lowright_x), 
                                                 'ImagePair':get_image_pair(df_save_sequence.loc[seq_index,'ImgNum'], seq_index, seq_num, seq_len), 
                                                 'Datetime':df_save_sequence.loc[seq_index,'Datetime'], 
                                                 'SeqNum':df_save_sequence.loc[seq_index,'SeqNum'], 
                                                 'SeqLen':df_save_sequence.loc[seq_index,'SeqLen'], 
                                                 'Night':0, 
                                                 'DiffMean':int(df_save_sequence.loc[seq_index,'DiffMean']), 
                                                 'DiffStd':int(df_save_sequence.loc[seq_index,'DiffStd']), 
                                                 'DiffMedian':int(df_save_sequence.loc[seq_index,'DiffMedian']), 
                                                 'NumObj':df_save_sequence.loc[seq_index,'NumObj'], 
                                                 'SizeMean':round(size_mean, 0), 
                                                 'SizeStd':round(size_std, 0), 
                                                 'SizeSkew':round(size_skew, 2), 
                                                 'DistMean':round(dist_mean, 0), 
                                                 'DistStd':round(dist_std, 0), 
                                                 'DistSkew':round(dist_skew, 2), 
                                                 'AngleMean':round(angle_mean, 0), 
                                                 'AngleStd':round(angle_std, 0), 
                                                 'AngleSkew':round(angle_skew, 2),
                                                 }, ignore_index=True)



               # Save all objects as detail records
               #for obj_index in range(0, nbr) :
               #    df_images = df_images.append({'File':df_save_sequence.loc[seq_index, 'File'], 'Dir':df_save_sequence.loc[seq_index,'Dir'], 'Datetime':df_save_sequence.loc[seq_index,'Datetime'], 'Camera':df_save_sequence.loc[seq_index,'Camera'], 'NumObj':df_save_sequence.loc[seq_index, 'NumObj'], 'DistRank':df_object.loc[obj_index, 'DistRank'], 'Size':df_object.loc[obj_index, 'Size'], 'X':df_object.loc[obj_index, 'X'], 'Y':df_object.loc[obj_index, 'X'], 'Dist':df_object.loc[obj_index, 'Dist'], 'Angle':df_object.loc[obj_index, 'Angle']}, ignore_index=True)
               df_object.drop(df_object.index[:], inplace=True)  # reinitialize for each pair in save_sequence
