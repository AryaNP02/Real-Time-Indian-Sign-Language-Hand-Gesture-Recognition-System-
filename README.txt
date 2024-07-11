######################################### INDIAN HAND SIGN DETECTION : FOR 26 ENGLISH ALPHABETS ###########################################################################################


+ HOW TO RUN  : 
 
   *** SET THE  BASE DIRECTORY  IN THIS VERY location i.e set ur base directory here  ***
              => this  is because , we have use  "relative paths" 
              => if wanted to make few changes  on the file location ,then remember to change the  "Paths" accordingly.


    * open the   "virtual environment" , activate it . [ in terminal :  {give the path to activate.ps1} ]; 
                   -   location  : venv - >  srcipts -> activate.ps1    
    
    **  give permission  to run the srcipts   , Only once  or always as per your wish ..

    *  RUN    :  app.py 



    ###### if for some reason  , the virtural environment  not working   the proced with following : 
    *  delete the  current  "venv" folder ,  then create ur own virtural environment  by : { command :: " python -m venv venv "}
    
    * install dependencies from "requirenments.txt"  :: {command :  pip install -r /path/to/requirements.txt }
                                                     ** if no changes in BASE DIRECTORY : { pip install -r requirements.txt }



   ##  directly  run :   " app.py "     ** may take time  when run for the first time  
                                        ** ignore any sort of warning   { it doesn't effect the performance of the model }

=============================================================================================================================================
PROJECT STRUCTURE 
     + app.py    = >  main file , execute this  file to start the app . 
      



    + Custom_models =>   we created a modules (  a folder ) , which stores everything need for the model .
                          ** it is adviced not to change  the   address of the files .. 
 
  
                            main files ={
       
                           
                                        two_Hand_ classifier    =>  whithin this : the weights of svm model are stored for
                                                                    2 hand classifier 

                                        one_Hand_ classifier   =>   within this,  the weights of knn  model  are stored  for
                                                                    1 hand classifier 
                                        }



    

==============================================================================================================================================
                          
 ++ COMMON ERRORS ::   
                          
    * The most common error   that could occur while  running the  "app.py "  will be : 
                "  directory not found "  or " could identified certain model " 
    

       SOLUTION ::  CHECK BASE DIRECTORY  or  CHECK BASE PATH ; 
                    also  IF THE NAME OF   "model weights file " IS CHANGED   -- > THEN CHECK THE NAME IN APP.PY.
                     
                     
                                                      
   