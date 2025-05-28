from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import cv2


class ImageProcessor:
    """
    cette classe permet de charger deux images, de détecter les points d'intérêt,
    de les relier entre eux puis de calculer la matrice fondamentale et la matrice essentielle.
    """
    
    def __init__(self, nom1:str, nom2:str) -> None:
        self.nom1: str = nom1
        self.nom2: str = nom2
        self.im1 = None
        self.im2 = None
        self.im1_gris = None
        self.im2_gris = None
        self.kp1 = None
        self.kp2 = None
        self.des1 = None
        self.des2 = None
        self.matches = None
        self.ratio_de_correspondance = 0.62
        self.sigma = 1.35
        self.octaveLayers = 2
        self.seuil_de_contraste = 0.050375
        self.seuil_d_aretes = 10
        self.max_features = 100

    # Load the image
    def load_images(self) -> None:
        """
        Charge les deux images à partir des noms de fichiers fournis.
        """
        try:
            image1 = cv2.imread(self.nom1)                
            image2 = cv2.imread(self.nom2)
            
            print(f"images correctement chargées :")
            print(f"{self.nom1} shape: {image1.shape} size:{image1.size}")
            print(f"{self.nom2} shape: {image2.shape} size:{image2.size}")
            
            self.im1 = image1
            self.im2 = image2
            
            self.im1_gris:cv2.typing.MatLike | None = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            self.im2_gris:cv2.typing.MatLike | None = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        
        except FileNotFoundError as e:
            print(f"EXCEPTION: {e}")            
        except Exception as e:
            print(f"EXCEPTION Unexpected error: {e}")
        
        
    def detect_sift_with_params(self) -> None:
            """Détecte les points SIFT avec paramètres personnalisés"""
            sift:Any = cv2.SIFT_create( # type: ignore
                nfeatures=self.max_features,
                nOctaveLayers=self.octaveLayers,
                contrastThreshold=self.seuil_de_contraste,
                edgeThreshold=self.seuil_d_aretes,
                sigma=self.sigma
            )
            
            # Détection des keypoints et descripteurs
            kp1, des1 = sift.detectAndCompute(self.im1_gris, None) # type: ignore
            kp2, des2 = sift.detectAndCompute(self.im2_gris, None) # type: ignore
            
            self.kp1 = kp1 # type: ignore
            self.kp2 = kp2 # type: ignore
            self.des1 = des1 # type: ignore
            self.des2 = des2 # type: ignore
            
    def affiche_images(self) -> None:
        # Affichage des images
        plt.figure(figsize=(13, 10)) # type: ignore
        plt.subplot(2, 2, 1) # type: ignore
        plt.imshow(cv2.cvtColor(self.im1, cv2.COLOR_BGR2RGB)) # type: ignore
        plt.title('Image 1') # type: ignore
        plt.axis('off') # type: ignore
        plt.subplot(2, 2, 2) # type: ignore
        plt.imshow(cv2.cvtColor(self.im2, cv2.COLOR_BGR2RGB)) # type: ignore
        plt.title('Image 2') # type: ignore
        plt.axis('off') # type: ignore
        plt.subplot(2, 2, 3) # type: ignore
        plt.imshow(cv2.cvtColor(self.im1_gris, cv2.COLOR_BGR2RGB)) # type: ignore
        plt.title('Image 2') # type: ignore
        plt.axis('off') # type: ignore
        plt.subplot(2, 2, 4) # type: ignore
        plt.imshow(cv2.cvtColor(self.im2_gris, cv2.COLOR_BGR2RGB)) # type: ignore
        plt.title('Image 2') # type: ignore
        plt.axis('off') # type: ignore
        plt.show() # type: ignore
        
    def afficheKP(self) -> None:
        # Affichage des keypoints
        plt.figure(figsize=(13, 10)) # type: ignore
        plt.subplot(2, 2, 1)
        image_kp_1 = cv2.drawKeypoints(self.im1_gris, self.kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite("image_kp_1.jpg", image_kp_1)
        plt.imshow(image_kp_1)
        plt.title('Image 1')
        plt.axis('off')
        plt.subplot(2, 2, 2)
        image_kp_2 = cv2.drawKeypoints(self.im2_gris, self.kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite("image_kp_2.jpg", image_kp_2)
        plt.imshow(image_kp_2)
        plt.title('Image 2')
        plt.axis('off')
        plt.show()
        
    def match_keypoints(self):
    
        """Apparie les descripteurs entre les deux images en utilisant FLANN matcher"""
            
        # FLANN matcher pour associer les descripteurs des deux images obtenus avec l'alglorithme SIFT
        FLANN_INDEX_KDTREE = 1
        index_params:cv2.typing.IndexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params:cv2.typing.SearchParams = dict(checks=100)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        try:
            matches = flann.knnMatch(self.des1, self.des2, k=2) # type: ignore
            
            good_matches: list[Any] = []
            cpt_paires_invalides = 0
            cpt_paires_valides = 0
                        
            # Test de ratio de Lowe
            for match_pair in matches: # type: ignore
                if len(match_pair) == 2: # type: ignore
                    cpt_paires_valides += 1
                    m , n = match_pair
                    if m.distance < self.ratio_de_correspondance * n.distance:
                        good_matches.append(m)
                else:
                    cpt_paires_invalides += 1
            print(f"Nombre de paires invalides : {cpt_paires_invalides} pour {cpt_paires_valides} paires valides et total de paires : {len(matches)}")
            self.matches = good_matches
        except cv2.error:
            self.matches = []
        
    def affiche_matches(self) -> None:
        # creer l'image des matches
        img_matches = cv2.drawMatches(self.im1, self.kp1, self.im2, self.kp2, # type: ignore
                                        self.matches, None, # type: ignore
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS) # type: ignore
        # Affichage de l'image des matches
        plt.figure(figsize=(13, 10))
        plt.imshow(img_matches)
        cv2.imwrite("matches.jpg", img_matches)
        plt.title('Matches')
        plt.axis('off')
        plt.show()
        
            

def main() -> None:
    ip = ImageProcessor("lit1.jpg", "lit3.jpg")
    ip.load_images()
    ip.detect_sift_with_params()
    ip.match_keypoints()
    ip.afficheKP()
    ip.affiche_matches()
    #ip.affiche_images()
    
main()
    
