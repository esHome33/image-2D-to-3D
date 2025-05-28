from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import cv2


class ImageProcessor:
    """
    cette classe permet de charger deux images, de détecter les points d'intérêt,
    de les relier entre eux puis de calculer la matrice fondamentale et la matrice essentielle.
    """
    
    def __init__(self) -> None:
        self.im1 = None
        self.im2 = None
        self.im1_gris = None
        self.im2_gris = None
        self.kp1 = None
        self.kp2 = None
        self.des1 = None
        self.des2 = None
        self.matches = None
        self.ratio_pour_matching = 0.654375

    # Load the image
    def load_images(self):
        """
        Load two images: new1.jpg and new2.jpg from the current folder
        
        Returns:
            tuple: (image1, image2) - Both images as numpy arrays in BGR format
        """
        try:
            image1 = cv2.imread('lit1.jpg')                
            image2 = cv2.imread('lit3.jpg')
            
            print(f"images correctement chargées :")
            print(f"new1.jpg shape: {image1.shape} size:{image1.size}")
            print(f"new2.jpg shape: {image2.shape} size:{image1.size}")
            
            self.im1 = image1
            self.im2 = image2
            
            self.im1_gris = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            self.im2_gris = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            
            return image1, image2
        
        except FileNotFoundError as e:
            print(f"EXCEPTION: {e}")
            return None, None
        except Exception as e:
            print(f"EXCEPTION Unexpected error: {e}")
            return None, None
        
        
    def detect_sift_with_params(self, nfeatures=0 , nOctaveLayers=6, contrastThreshold=0.05, 
                                edgeThreshold=10, sigma=1.4)-> tuple[Any, Any, Any, Any]:
            """Détecte les points SIFT avec paramètres personnalisés"""
            sift = cv2.SIFT_create(
                nfeatures=nfeatures,
                nOctaveLayers=nOctaveLayers,
                contrastThreshold=contrastThreshold,
                edgeThreshold=edgeThreshold,
                sigma=sigma
            )
            
            # Détection des keypoints et descripteurs
            kp1, des1 = sift.detectAndCompute(self.im1_gris, None)
            kp2, des2 = sift.detectAndCompute(self.im2_gris, None)
            
            self.kp1 = kp1
            self.kp2 = kp2
            self.des1 = des1
            self.des2 = des2
            
    def affiche_images(self) -> None:
        # Affichage des images
        plt.figure(figsize=(13, 10))
        plt.subplot(2, 2, 1)
        plt.imshow(cv2.cvtColor(self.im1, cv2.COLOR_BGR2RGB))
        plt.title('Image 1')
        plt.axis('off')
        plt.subplot(2, 2, 2)
        plt.imshow(cv2.cvtColor(self.im2, cv2.COLOR_BGR2RGB))
        plt.title('Image 2')
        plt.axis('off')
        plt.subplot(2, 2, 3)
        plt.imshow(cv2.cvtColor(self.im1_gris, cv2.COLOR_BGR2RGB))
        plt.title('Image 2')
        plt.axis('off')
        plt.subplot(2, 2, 4)
        plt.imshow(cv2.cvtColor(self.im2_gris, cv2.COLOR_BGR2RGB))
        plt.title('Image 2')
        plt.axis('off')
        plt.show()
        
    def afficheKP(self) -> None:
        # Affichage des keypoints
        plt.figure(figsize=(13, 10))
        plt.subplot(2, 2, 1)
        plt.imshow(cv2.drawKeypoints(self.im1_gris, self.kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
        plt.title('Image 1')
        plt.axis('off')
        plt.subplot(2, 2, 2)
        plt.imshow(cv2.drawKeypoints(self.im2_gris, self.kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
        plt.title('Image 2')
        plt.axis('off')
        plt.show()
        
    def match_keypoints(self):
    
        """Apparie les descripteurs entre les deux images en utilisant FLANN matcher"""
            
        # FLANN matcher pour associer les descripteurs des deux images obtenus avec l'alglorithme SIFT
        FLANN_INDEX_KDTREE = 1
        index_params:cv2.typing.IndexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=8)
        search_params:cv2.typing.SearchParams = dict(checks=100)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        try:
            matches:Any = flann.knnMatch(self.des1, self.des2, k=2)
            
            good_matches: list[Any] = []
            cpt_paires_invalides = 0
            cpt_paires_valides = 0
                        
            # Test de ratio de Lowe
            for match_pair in matches:
                if len(match_pair) == 2:
                    cpt_paires_valides += 1
                    m , n = match_pair
                    if m.distance < self.ratio_pour_matching * n.distance:
                        good_matches.append(m)
                else:
                    cpt_paires_invalides += 1
            print(f"Nombre de paires invalides : {cpt_paires_invalides} pour {cpt_paires_valides} paires valides et total de paires : {len(matches)}")
            self.matches = good_matches
        except cv2.error:
            self.matches = []
        
    def affiche_matches(self) -> None:
        # creer l'image des matches
        img_matches = cv2.drawMatches(self.im1, self.kp1, self.im2, self.kp2, 
                                        self.matches, None, 
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # Affichage de l'image des matches
        plt.figure(figsize=(13, 10))
        plt.imshow(img_matches)
        cv2.imwrite("matches.jpg", img_matches)
        plt.title('Matches')
        plt.axis('off')
        plt.show()
        
            

def main() -> None:
    ip = ImageProcessor()
    ip.load_images()
    ip.detect_sift_with_params()
    ip.match_keypoints()
    ip.affiche_matches()
    #ip.affiche_images()
    #ip.afficheKP()
    
main()
    
