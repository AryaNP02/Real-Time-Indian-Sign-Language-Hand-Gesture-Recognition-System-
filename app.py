# these libraries help to do certain tasks  ..


import copy  
import itertools 
import numpy as np


import warnings
warnings.filterwarnings("ignore")



# main libraries 
import cv2 as cv  
import joblib
import mediapipe as mp
 




## Note :  below are relative paths, therby  check ur  base directory   and accordingly  set the paths
   
# for 2 hand 
svm_model_path = r'Custom_models/keypoint_classifier/two_Hand_ classifier/best_model2.pkl'

 #for 1  hand  

knn_1_hand_model1 =  r'Custom_models/keypoint_classifier/one_hand_classifier/knn_model_for_1_hand_0.99.joblib'

###################################################################################################################


####################  gesture labels #####################
keypoint_classifier_labels = ['A', 'B', 'D', 'E', 'F', 'G', 'H', 'K', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'W', 'X',
                               'Y', 'Z', 'C', 'I', 'J', 'L', 'O', 'U', 'V']

###########################################################


def main():
    # ############################################################################################################
    # here we define the parameter that will be used for : camera configuration   & Mediapipe
   
    
    # for camera 
    cap_device = 0   ## by this we choose the video capturing device ( camera) ; 0 -> denotes to use default camera of ur device
    cap_width = 960   ##  by these , we define the dimension  i.e height and width of  display window ( frame )
    cap_height = 540

    ## for setting  Mediapipe  ---  it is a frameWork  using which we detect and  trace the movement of hands. 

    no_of_hands = 2 
    min_detection_confidence = 0.7
    min_tracking_confidence = 0.5
   

    

    # Camera preparation #######################################################################################
     # here we  set our camera .. 
    
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # setup mediapipe for hands detection #######################################################################

    mp_hands = mp.solutions.hands   ## mediapipe .. "hand" detection parameter initialization . 

    hands = mp_hands.Hands(
        
        max_num_hands= no_of_hands, # max hands to be tracked 
        min_detection_confidence=min_detection_confidence,
        #  set  minimum value for detection confidence i.e  hands will be dectected when the model confidence
        #  is above certain value. 
        min_tracking_confidence=min_tracking_confidence, 
        
        # sets minimum value for successfull  tracking  . 
    )

    ############################################################################################################
    # two hand models  ... here we created instances .. 

    
    svm_model = joblib.load(svm_model_path)
   

    # one hand models .... created instance

    onehand_knn_model = joblib.load(knn_1_hand_model1)
    #############################################################################################################








    #  ######################################################################################
    

    while True:
       

        
        # PRESS ESC  KEY TO TERMINATE / CLOSE THE WEBCAM 
        key = cv.waitKey(10)  
        if key == 27:  # ESC
            break
        

        # Camera capture ####################################################################
        #  note : 'err' variable is to detect any "error" occur , while reading the webcam
        # image is our frame.  

        noerr, image = cap.read()  

        # if error occur , terminate the process . 
        if not noerr:
            break

        # HORIZONTALLY FLIP THE IMAGE  ..
        image = cv.flip(image, 1) # remove mirror image view .

        copiedImage = copy.deepcopy(image)

        # Detection implementation #############################################################
        #   by default the cv2 processing format is : BGR ( blue green  red ) , but mediapipe RGB ( red green blue)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    

        
        results = hands.process(image)
       



        #  ###################################################################################33
        if results.multi_hand_landmarks is not None:  ##  check if hand detected  
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):   ## loop through landmarks .. 
               
               
                # Bounding box calculation
                brect = calc_bounding_rect(copiedImage, hand_landmarks)



                # Landmark calculation
                # STORE THE LANDMARKS  into a list 

                pre_processed_landmark_list=[]
            
                
                for hand_landmark  in  results.multi_hand_landmarks:
                    landmark_list = calc_landmark_list(copiedImage, hand_landmark)

                    # Conversion to relative coordinates , normalized coordinates
                    pre_processed_landmark_list .append( pre_process_landmark(
                        landmark_list))
                    


                # Check hand :  if len ( list )  > 1 : shows two hand detected 
                # if 2 hands :: combine them
                 # else     use the list as it is  
                    
                if len(pre_processed_landmark_list) > 1:
                    pre_processed_landmark_list= pre_processed_landmark_list[0] + pre_processed_landmark_list[1]
                else:
                    pre_processed_landmark_list=pre_processed_landmark_list[0]



                #To see how  a  processed -landmark  look after concatenation   ;; uncomment the below to 2  line .
                
                #print(pre_processed_landmark_list)
                #print(len(pre_processed_landmark_list))
                    
        ############################### $$$  CLASSIFICATION CODE  $$$ ##############################################################################
                    
                FromModel =""

                if len(pre_processed_landmark_list) == 84:

                    
                    hand_sign_id = svm_model.predict([pre_processed_landmark_list])[0]


                    FromModel="Two-handed Gesture "

                    #print(hand_sign_id)
                else:
                    

                    hand_sign_id = onehand_knn_model.predict([pre_processed_landmark_list])[0]+19


                    FromModel="One-handed Gesture "




    
####################################################### classification result index stored in hand_sign_id ########################
                # Drawing part :  below draw the box around your  hands 
                copiedImage = draw_bounding_rect(copiedImage , brect)
                
               
                copiedImage = draw_info_text(
                    copiedImage,
                    brect, 
                    handedness,
                    keypoint_classifier_labels[hand_sign_id], 
            
                    
                    FromModel  
                  
                )
#####################################################################################################################################


  

        # Screen  #############################################################
        cv.imshow('Hand Gesture Recognition', copiedImage)  


    cap.release()
    cv.destroyAllWindows()
##########################################################  END OF MAIN FUNCTION #####################################


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)
    #print( landmark_array.shape)
    

    for _, landmark in enumerate(landmarks.landmark):
        
        landmark_x = min(int(landmark.x * image_width), image_width - 1)  #  ensure the points within image range.
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]
    

        landmark_array = np.append(landmark_array, landmark_point, axis=0)
    #print(landmark_array)

    x, y, w, h = cv.boundingRect(landmark_array) ## func :  get bounding box    return ->  start point (x,y) ; width $ height  

    return [x, y, x + w, y + h]
######################################################################################################################

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)# ensure point  stays with the frame size 
        landmark_y = min(int(landmark.y * image_height), image_height - 1)# ensure point  stays with the frame size 


        landmark_point.append([landmark_x, landmark_y])

    return landmark_point   # we  use only x, y landmarks ...   and not z landmarks ( trace how close and away
                                                                                        #our hand is from cam.  )


######################################################################################################################
def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list) # copy everything  - content 

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))  # absoulte value  : because relative points  can be negative.   

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list





#########################################################################################################################################

def draw_bounding_rect( image, brect):
    
        # Outer rectangle ,draws bracket onto the frame  
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image

#########################################################################################################################################
def draw_info_text(image, brect, handedness, hand_sign_text, model_name):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)   # frame , diagonal points  , color , fill or not fill  .

    info_text = handedness.classification[0].label[0:]  # determine hand is   : left  or right . 
    if hand_sign_text != "":  ##  if model gives an perdiction . 
        info_text = info_text + ':' + hand_sign_text
        cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
        

    # Draw "Hand Gesture" text at the top-left corner with double boundary
    cv.putText(image, "Hand Gesture", (20, 40),
               cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3, cv.LINE_AA)
    cv.putText(image, "Hand Gesture", (20, 40),
               cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv.LINE_AA)

    # Draw model name and corresponding hand sign below "Hand Gesture" text with double boundary
    text = model_name + ": " + hand_sign_text
    cv.putText(image, text, (20, 90),
               cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv.LINE_AA)
    cv.putText(image, text, (20, 90),
               cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)


    return image

#########################################################################################################################################
#  calling main -  Program execution start from   here.
#########################################################################################################################################
if __name__ == '__main__':
    
    main()
