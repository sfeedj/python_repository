# -*- coding: utf-8 -*-
import cv2
import numpy as np
from math import sqrt
import serial 



# INITIALISATION 


cap = cv2.VideoCapture(1)

j=None
cmdMode=False
clrMode=False
clrobsMode = False
commande, deplacement = [],[]
frame = None
center=[]
front, front_init = None, None
rect_obstacles=[]
box_obstacles=[]

port = serial.Serial('COM4',baudrate=9600, timeout=1)
entree = serial.Serial('COM5',baudrate=9600, timeout=0)

lower_clr = np.array([90,30,30])
upper_clr= np.array([110,245,245]) # intervalle à régler sur la couleur voulue (hsv) grâce à la fonction clr. bleu par défaut.

lower_clrobs = np.array([0,0,0])
upper_clrobs= np.array([130,110,40])

# DECLARATION DES DEUX FONCTIONS : CMD (commande) ET CLR (couleur)


def cmd(event, x, y, flag, param):

    global frame, commande, cmdMode  
 
    if cmdMode and event == cv2.EVENT_LBUTTONUP :
        clrMode=False #
        clrobsMode = False
        commande = [x,y]
        cv2.putText(frame, 'Cmd Sent',(30,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),1)
        print('commande :', commande)




def clr(event, x, y,flag, param):
    
    global frame, lower_clr, upper_clr
    
    if clrMode and event==cv2.EVENT_LBUTTONUP:
        cmdMode= False
        clrobsMode = False
        px=frame[y,x]
        
        px_bgr=np.uint8([[px]])
        
        px_hsv=cv2.cvtColor(px_bgr,cv2.COLOR_BGR2HSV)
        
        lower_clr = np.asarray([x + y for x, y in zip(px_hsv[0][0],[-15,-60,-30])])
        upper_clr = np.asarray([x + y for x, y in zip(px_hsv[0][0],[15,60,30])]) 
        
        
        
        #NOTE : l'ordre y,x est CRUCIAL
def clrobs(event, x, y,flag, param):
    
    global frame, lower_clrobs, upper_clrobs
    
    if clrobsMode and event==cv2.EVENT_LBUTTONUP:
        cmdMode = False
        clrMode = False
        pxo=frame[y,x]
        
        px_bgro=np.uint8([[pxo]])
        
        px_hsvo=cv2.cvtColor(px_bgro,cv2.COLOR_BGR2HSV)
        
        lower_clrobs = np.asarray([x + y for x, y in zip(px_hsvo[0][0],[-15,-40,-40])])
        upper_clrobs = np.asarray([x + y for x, y in zip(px_hsvo[0][0],[15,40,40])]) 
        print (lower_clrobs)
        
cv2.namedWindow("frame")


#DEBUT DU TRAITEMENT D'IMAGE 
    
while True :
    
    _,frame = cap.read()
    
#GESTION DES MASQUES
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # converti en hsv
    blur= cv2.GaussianBlur(hsv, (5,5), 5) # floute l'image
    
    #VOITURE
    mask = cv2.inRange(blur, lower_clr, upper_clr) # creation du mask : blanc pour les pixel de couleur dans l'intervalle, noir sinon    
    mask= cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))) #réalisation d'un opening sur le masque    
    contours=cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1] #trouve les contours du masque
    
    cv2.imshow('mask',mask)
    
    #OBSTACLES
    mask_obstacle = cv2.inRange(blur, lower_clrobs, upper_clrobs)
    mask_obstacle= cv2.morphologyEx(mask_obstacle, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))) #réalisation d'un opening sur le masque
    contours_obstacles=cv2.findContours(mask_obstacle.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1] #trouve les contours du masque
   
    cv2.imshow('mask obstacle',mask_obstacle)
    
        
    
# GESTION DES CONTOURS ET CREATION DE LA ROI (Region Of Interest) RECTANGULAIRE    
    
    if len(contours) !=0: #si un contour est détecté : (évite les crash si la couleur à détecter sort du champ de la caméra)
        
        cmax = max(contours, key=cv2.contourArea) # si des contours existent, alors cmax = le plus grand d'entre eux 
    
        rect = cv2.minAreaRect(cmax) # retourne un tuple ( center (x,y), (largeur, hauteur), angle de rotation bluetooth      
        box = np.int0(cv2.boxPoints(rect)) #retourne les 4 sommets du rectangle ; Arrondie chaque terme � l'entier pr�t
        cv2.drawContours(frame,[box],0,(0,0,255),2) # creation d'un rectangle d'aire minimum autour de cmax
                
        center = (int(rect[0][0]), int(rect[0][1])) 
        
        if j not in [center,(int(rect[0][0])+1, int(rect[0][1])+1), (int(rect[0][0])-1, int(rect[0][1])-1), (int(rect[0][0]), int(rect[0][1])-1),(int(rect[0][0]), int(rect[0][1])+1), (int(rect[0][0])-1, int(rect[0][1])),(int(rect[0][0])+1, int(rect[0][1])),(int(rect[0][0])+1, int(rect[0][1])-1), (int(rect[0][0])-1, int(rect[0][1])+1) ] :
            #intervalle de fluctuation pour stabiliser le centre

            cv2.circle(frame,center, 3, 3, thickness=2, lineType=8, shift=0) # on en dessine le centre et on recupere ses coordonnees .
            #print ('center coord :', center)
            j=center # on calcul et affiche le centre s'il a changé, sinon non.
        
        else:
            cv2.circle(frame,j, 3, 3, thickness=2, lineType=8, shift=0)   # on en dessine le centre et on recupere ses coordonnées .
                

        A,B,C,D = box[0],box[1],box[2],box[3] # On extrait du rectangle d'interêt les sommets A,B,C,D. A est toujours le points le plus bas, puis on tourne dans le sens horaire. 
        AB=int(sqrt((B[0]-A[0])**2+(B[1]-A[1])**2))
        BC=int(sqrt((C[0]-B[0])**2+(C[1]-B[1])**2)) # On calcule les longueurs AB, BC, etc...
        CD=int(sqrt((D[0]-C[0])**2+(D[1]-C[1])**2))  
        DA=int(sqrt((A[0]-D[0])**2+(A[1]-D[1])**2))
        
        cv2.putText(frame,'A',tuple(A),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),1 ) #affiche le point 'A'
        cv2.putText(frame,'B',tuple(B),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),1 ) #affiche le point 'B'
        cv2.putText(frame,'C',tuple(C),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),1 ) #affiche le point 'C'
        cv2.putText(frame,'D',tuple(D),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),1 ) #affiche le point 'D'
        
        
# DETECTION DE L'AVANT DE L'OBJET : CONDITIONS SUR LES ROTATIONS DU RECTANGLE
        
 
        if AB<BC: 
            front = np.int0(0.5*(np.array(A)+np.array(B)))

        else :
            front = np.int0(0.5*(np.array(B)+np.array(C))) #BC
         
         
   
        if front_init != None:
            if abs(front[1]-front_init[1])>10:
                
                if (front[1]-front_init[1])>0:             
                    front = np.int0(0.5*(np.array(C)+np.array(D)))
 
                else :
                    front = np.int0(0.5*(np.array(A)+np.array(D)))

            elif abs(front[0]-front_init[0])>30:
                if (front[0]-front_init[0])>0 :
                    front = np.int0(0.5*(np.array(C)+np.array(D)))   
   
                else:
                    front = np.int0(0.5*(np.array(A)+np.array(D))) 
 
        front_init=front
     
        cv2.circle(frame,tuple(front), 3, 3, thickness=2, lineType=8, shift=0)
         
        cv2.line(frame, tuple(front), center, (255,0,0), thickness=1, lineType=8, shift=0)
        
#AFFICHAGE DE LA COMMANDE      
        if commande != [] and cmdMode==True: #on affiche le point correspondant à la commande
            commande_t=tuple(commande)
            cv2.circle(frame,commande_t, 3, 3, thickness=2, lineType=8, shift=0)
            cv2.line(frame, center, commande_t, 3, thickness=1, lineType=8, shift=0)
                

            
            
            
# GESTION DES OBSTACLES : TRACES ET INTERSECTION
        if len(contours_obstacles) > 0:
                        
                        
            cmax_obstacles = max(contours_obstacles, key=cv2.contourArea) # si des contours existent, alors cmax = le plus grand d'entre eux 
            rect_obstacles = cv2.minAreaRect(cmax_obstacles) # retourne un tuple ( center (x,y), (largeur, hauteur), angle de rotation bluetooth      
            box_obstacles = np.int0(cv2.boxPoints(rect_obstacles)) #retourne les 4 sommets du rectangle ; Arrondie chaque terme à l'entier prêt
            cv2.drawContours(frame,[box_obstacles],0,(0,0,255),2) # creation d'un rectangle d'aire minimum autour de cmax
                
            As,Bs,Cs,Ds = box_obstacles[0],box_obstacles[1],box_obstacles[2],box_obstacles[3] # On extrait du rectangle d'interêt les sommets A,B,C,D. A est toujours le points le plus bas, puis on tourne dans le sens horaire. 
            
            AsBs=int(sqrt((Bs[0]-As[0])**2+(Bs[1]-As[1])**2))
            BsCs=int(sqrt((Cs[0]-Bs[0])**2+(Cs[1]-Bs[1])**2))
            centre_obstacle = rect_obstacles[0]
            taille_obstacle = int(sqrt((centre_obstacle[0]-As[0])**2+(centre_obstacle[1]-As[1])**2)) 
            
            if commande != []:
                
                taille_voiture = 2*int(sqrt((front[0]-center[0])**2+(front[1]-center[1])**2)) 
                distance_commande = int(sqrt((front[0]-commande[0])**2+(front[1]-commande[1])**2))   
                
                #####
                def segment(p1, p2):
                    a = (p1[1] - p2[1])
                    b = (p2[0] - p1[0])
                    c = -(p1[0]*p2[1] - p2[0]*p1[1])
                    return a, b, c
            
            
                def intersection(segment1, segment2, commande):
                    global front
                    D  = segment1[0] * segment2[1] - segment1[1] * segment2[0]
                    Dx = segment1[2] * segment2[1] - segment1[1] * segment2[2]
                    Dy = segment1[0] * segment2[2] - segment1[2] * segment2[0]

                    if D != 0 : 
                        x = Dx / D
                        y = Dy / D
                        pt_intersect=[x,y]
                        distance_commande = int(sqrt((front[0]-commande[0])**2+(front[1]-commande[1])**2))
                        distance_intersect = int(sqrt((front[0]-pt_intersect[0])**2+(front[1]-pt_intersect[1])**2))
                        distance_intersect_obstacle = int(sqrt(abs((centre_obstacle[0]-pt_intersect[0])**2+(centre_obstacle[1]-pt_intersect[1])**2)))
                        if distance_commande > distance_intersect and distance_intersect_obstacle < taille_obstacle  :
                            return pt_intersect
                        else: return False
                    else:
                        return False
                    #####
                    
                segment_cmd = segment(center, commande)
                
                if AsBs>=BsCs:
                    segment_obstacle = segment(As, Bs)
                else : 
                    segment_obstacle = segment(Bs, Cs)
                    
                pt_intersect = intersection(segment_cmd, segment_obstacle,commande)


                    
        # DETECTION DE LA ROTATION COMMANDEE
                    
        
    
                points = np.array([center, front, commande]) #important de convertir en array, pour la compatibilité avec la soustraction.
                s1 = points[1] - points[0]
                s2 = points[0] - points[2] 
                num = np.dot(s2, -s1)
                denom = np.linalg.norm(s2) * np.linalg.norm(-s1)
                angle_commande=int((np.arccos(num/denom) * 180 / np.pi))
                print angle_commande
                
                cv2.putText(frame,str(angle_commande),center,cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)
            
                if pt_intersect != False:
                    print pt_intersect
                    cv2.circle(frame,tuple(pt_intersect), 3, (0,0,255), thickness=2, lineType=8, shift=0)
                    sous_commande = [As[0]-(taille_voiture), As[1]-(taille_voiture//3)]
                    cv2.line(frame, center, tuple(sous_commande), (0,255,0), thickness=2, lineType=8, shift=0)
                    cv2.line(frame, tuple(sous_commande), commande_t, (0,255,0), thickness=2, lineType=8, shift=0)
                    
                    points = np.array([center, front, sous_commande]) #important de convertir en array, pour la compatibilité avec la soustraction.
                    s1 = points[1] - points[0]
                    s2 = points[0] - points[2] 
                    numS = np.dot(s2, -s1)
                    denomS = np.linalg.norm(s2) * np.linalg.norm(-s1)
                    angle_sous_commande=int((np.arccos(numS/denomS) * 180 / np.pi))
                    distance_sous_commande = int(sqrt((front[0]-sous_commande[0])**2+(front[1]-sous_commande[1])**2))

                    print angle_sous_commande        
                    
                    if(sous_commande[0]<center[0]):
                        b=angle_sous_commande+256
                        b=str(b)

                    elif (sous_commande[0]>center[0]):
                            b=360-angle_sous_commande +256
                            b=str(b)
                    port.write(b)
                    if (angle_sous_commande<30):
                        port.write('150')
                        pass
                                      
                    if distance_sous_commande < taille_voiture:
                        port.write('000')
                        pass

                    
                distance_commande = int(sqrt((front[0]-commande[0])**2+(front[1]-commande[1])**2))
                if(commande[0]<center[0]):
                    b=angle_commande+256
                    b=str(b)
    
                elif (commande[0]>center[0]):
                        b=360-angle_commande +256
                        b=str(b)
                        port.write(b)
                if (angle_commande<30):
                    port.write('150')
                    pass
                
    
                      
                if distance_commande < taille_voiture:
                    port.write('000')
                    pass
               
                else:
                    port.write('150') # avancée de la voiture
                    pass
                    # on trace la perpandiculaire de la trajectoire passant par le point de la commande
                    # dès qu'il y a intersection entre cette perpandiculaire et le front de la voiture, on stoppe la voiture 


           


        
# GESTION DES ENTREES CLAVIER / INTERFACE UTILISATEUR   
         
    k = cv2.waitKey(1) & 0xFF #on stock l'entrée clavier dans k
        
    if k == 27:
        cv2.destroyAllWindows()
        break # on quitte en appuyant sur échape
        
    elif k == ord('a') :
        cmdMode=True # entrée en mode commande
        cv2.putText(frame, 'CmdMode On',(30,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),1)
        cv2.setMouseCallback("frame", cmd) # On rattache la fonction cmd à la fenetre de la video 

            
    elif k == ord('q'): 
        cmdMode = False 
        cv2.putText(frame, 'CmdMode Off',(30,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),1)

        
    elif k == ord('p'):
        clrMode=True
        cv2.putText(frame, 'ClrMode On',(30,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),1)
        cv2.setMouseCallback("frame",clr)


        
    elif k == ord('m'): #quitte le mode couleur
        clrMode = False
        cv2.putText(frame, 'ClrMode Off',(30,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),1)
    
    elif k == ord('w'):
        clrobsMode=True
        cv2.putText(frame, 'ClrobsMode On',(30,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),1)
        cv2.setMouseCallback("frame",clrobs)
    
    elif k == ord('x'):
        clrobsMode=False
        

    
#FENETRES OUVERTES :


    cv2.imshow('frame',frame)
    
#     cv2.imshow('blur',blur)
#     cv2.imshow('res',res)
#     cv2.imshow("hsv",blur)