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
        """Initialise cette classe avec les noms des deux images à charger.

        Args:
            nom1 (str): nom de l'image 1 (avec chemin d'accès si nécessaire)
            nom2 (str): nom de l'image 2 (avec chemin d'accès si nécessaire)
        """
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
        
        self.homographie = None

    # Load the image
    def load_images(self) -> None:
        """
        Charge les deux images à partir des noms de fichiers fournis.
        """
        try:
            image1 = cv2.imread(self.nom1)                
            image2 = cv2.imread(self.nom2)
            
            print(f"images correctement chargées :")
            print(f"{self.nom1} shape: {image1.shape} size:{image1.size}") #type: ignore
            print(f"{self.nom2} shape: {image2.shape} size:{image2.size}") #type: ignore
            
            self.im1 = image1
            self.im2 = image2
            
            self.im1_gris:cv2.typing.MatLike | None = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            self.im2_gris:cv2.typing.MatLike | None = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        
        except FileNotFoundError as e:
            print(f"EXCEPTION: {e}")            
        except Exception as e:
            print(f"EXCEPTION Unexpected error: {e}")
        
        
    def detecte_keypoints_avec_SIFT(self) -> None:
            """Détecte les keypoints avec l'algorithme SIFT et des paramètres personnalisés"""
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
        """
        Affiche les deux images (en couleur et en gris)
        """
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
        """
        Affiche les keypoints de chaque image dans une fenêtre
        """
        # Affichage des keypoints
        plt.figure(figsize=(13, 10)) # type: ignore
        plt.subplot(2, 2, 1) # type: ignore
        image_kp_1 = cv2.drawKeypoints(self.im1_gris, self.kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) # type: ignore
        cv2.imwrite("image_kp_1.jpg", image_kp_1) # type: ignore
        plt.imshow(image_kp_1) # type: ignore
        plt.title('Image 1') # type: ignore
        plt.axis('off') # type: ignore
        plt.subplot(2, 2, 2) # type: ignore
        image_kp_2 = cv2.drawKeypoints(self.im2_gris, self.kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) # type: ignore
        cv2.imwrite("image_kp_2.jpg", image_kp_2) # type: ignore
        plt.imshow(image_kp_2) # type: ignore
        plt.title('Image 2') # type: ignore
        plt.axis('off') # type: ignore
        plt.show() # type: ignore
        
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
                    m , n = match_pair #type: ignore
                    if m.distance < self.ratio_de_correspondance * n.distance:  # type: ignore
                        good_matches.append(m)
                else:
                    cpt_paires_invalides += 1
            print(f"Nombre de paires invalides : {cpt_paires_invalides} pour {cpt_paires_valides} paires valides et total de paires : {len(matches)}") #type: ignore
            self.matches = good_matches
        except cv2.error:
            self.matches = []
        
    def affiche_matches(self) -> None:
        # creer l'image des matches
        img_matches = cv2.drawMatches(self.im1, self.kp1, self.im2, self.kp2, # type: ignore
                                        self.matches, None, # type: ignore
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS) # type: ignore
        # Affichage de l'image des matches
        plt.figure(figsize=(13, 10)) # type: ignore
        plt.imshow(img_matches) # type: ignore
        cv2.imwrite("matches.jpg", img_matches) # type: ignore
        plt.title('Matches') # type: ignore
        plt.axis('off') # type: ignore
        plt.show() # type: ignore
      
    def calcule_homographie(self) -> None:
        """
        Calcule l'homographie entre les deux images
        """
        # Récupération des points correspondants
        pts1 = np.float32([self.kp1[m.queryIdx].pt for m in self.matches]).reshape(-1, 1, 2) # type: ignore
        pts2 = np.float32([self.kp2[m.trainIdx].pt for m in self.matches]).reshape(-1, 1, 2) # type: ignore

        # Calcul de l'homographie
        self.homographie, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0) # type: ignore
        print("homographie ",self.homographie)
    
    
    def calc_matrice_essentielle(self):# type: ignore
        """
        Trouve la matrice essentielle et triangule les points 3D
        
        Utilise:
            kp1, kp2: Keypoints des deux images (trouvés avec detecte_keypoints_avec_SIFT())
            matches : les correspondances entre les deux images (trouvées avec match_keypoints())
        
        Renvoie:
            points_3d: Points 3D triangulés
            matched_points1, matched_points2: Points correspondants dans les deux images
            E: Matrice essentielle
            R, t: Rotation et translation entre les deux caméras
        """
        
        # Matrice de calibrage d'une caméra standard        
        K:np.typing.ArrayLike = np.array([
            [800, 0, 320],
            [0, 800, 240],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Filtrage des correspondances
        if len(self.matches) < 8: # type: ignore
            raise ValueError("Pas assez de correspondances pour calculer la matrice essentielle (minimum 8)")
        
        # Extraction des points correspondants
        pts1 = np.float32([self.kp1[m.queryIdx].pt for m in self.matches]).reshape(-1, 1, 2) # type: ignore
        pts2 = np.float32([self.kp2[m.trainIdx].pt for m in self.matches]).reshape(-1, 1, 2) # type: ignore
        
        # Calcul de la matrice essentielle avec RANSAC
        E, mask = cv2.findEssentialMat(
            pts1, pts2, K, 
            method=cv2.RANSAC, 
            prob=0.999, 
            threshold=1.0
        )
        
        # Filtrer les points avec le masque RANSAC
        pts1_inliers = pts1[mask.ravel() == 1]
        pts2_inliers = pts2[mask.ravel() == 1]
        
        print(f"Nombre d'inliers après RANSAC: {len(pts1_inliers)}")
        
        # Récupération de la pose (rotation et translation) à partir de la matrice essentielle
        _, R, t, mask_pose = cv2.recoverPose(E, pts1_inliers, pts2_inliers, K) #type: ignore
        
        # Calcul des matrices de projection
        # Caméra 1 (référence)
        P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        
        # Caméra 2 (avec rotation et translation) ... on prend la même matrice de calibrage que pour la caméra 1
        P2 = K @ np.hstack([R, t])
        
        # Triangulation des points 3D
        # Normalisation des points pour la triangulation
        pts1_norm = cv2.undistortPoints(pts1_inliers, K, None) #type: ignore
        pts2_norm = cv2.undistortPoints(pts2_inliers, K, None) #type: ignore
        
        # Triangulation
        points_4d = cv2.triangulatePoints(P1, P2, pts1_norm, pts2_norm) #type: ignore
        
        # Conversion en coordonnées 3D homogènes dans le repère de la caméra 1
        points_3d = points_4d[:3] / points_4d[3]  #type: ignore
        points_3d = points_3d.T  #type: ignore
        
        return {  #type: ignore
            'points_3d': points_3d,
            'matched_points1': pts1_inliers,
            'matched_points2': pts2_inliers,
            'essential_matrix': E,
            'rotation': R,
            'translation': t,
            'projection_matrix1': P1,
            'projection_matrix2': P2
        }
    
    
    def plot_3d_points_with_cameras(self, points_3d, R, t, title="Points 3D avec positions des caméras"): # type: ignore
        """
        Trace les points 3D avec les positions et orientations des caméras
        
        Args:
            points_3d: Points 3D reconstruits
            R: Matrice de rotation de la caméra 2
            t: Vecteur de translation de la caméra 2
            title: Titre du graphique
        """
        fig = plt.figure(figsize=(14, 10)) # type: ignore
        ax = fig.add_subplot(111, projection='3d') # type: ignore
        
        # Points 3D
        x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2] # type: ignore
        ax.scatter(x, y, z, c='blue', s=30, alpha=0.7, label='Points 3D') # type: ignore
        
        # Position de la caméra 1 (origine)
        ax.scatter([0], [0], [0], c='red', s=100, marker='^', label='Caméra 1') # type: ignore
        
        # Position de la caméra 2
        cam2_pos = -R.T @ t # type: ignore
        ax.scatter([cam2_pos[0, 0]], [cam2_pos[1, 0]], [cam2_pos[2, 0]], c='green', s=100, marker='^', label='Caméra 2') # type: ignore
        
        # Axes de la caméra 1 (origine)
        axis_length = 2.0 # type: ignore
        ax.quiver(0, 0, 0, axis_length, 0, 0, color='red', alpha=0.6, arrow_length_ratio=0.1) # type: ignore
        ax.quiver(0, 0, 0, 0, axis_length, 0, color='green', alpha=0.6, arrow_length_ratio=0.1) # type: ignore
        ax.quiver(0, 0, 0, 0, 0, axis_length, color='blue', alpha=0.6, arrow_length_ratio=0.1) # type: ignore
        
        # Axes de la caméra 2
        R_axes = R.T  # Rotation inverse pour les axes # type: ignore
        cam2_x = cam2_pos.flatten() + R_axes[:, 0] * axis_length # type: ignore
        cam2_y = cam2_pos.flatten() + R_axes[:, 1] * axis_length # type: ignore
        cam2_z = cam2_pos.flatten() + R_axes[:, 2] * axis_length # type: ignore
        
        ax.plot3D([cam2_pos[0, 0], cam2_x[0]], [cam2_pos[1, 0], cam2_x[1]], [cam2_pos[2, 0], cam2_x[2]],  # type: ignore
                'r--', alpha=0.6)
        ax.plot3D([cam2_pos[0, 0], cam2_y[0]], [cam2_pos[1, 0], cam2_y[1]], [cam2_pos[2, 0], cam2_y[2]],  # type: ignore
                'g--', alpha=0.6)
        ax.plot3D([cam2_pos[0, 0], cam2_z[0]], [cam2_pos[1, 0], cam2_z[1]], [cam2_pos[2, 0], cam2_z[2]],  # type: ignore
                'b--', alpha=0.6)        

        # Configuration des axes
        ax.set_xlabel('X') # type: ignore
        ax.set_ylabel('Y') # type: ignore
        ax.set_zlabel('Z') # type: ignore
        ax.set_title(title) # type: ignore
        ax.legend() # type: ignore
        
        # Égalisation des échelles
        all_points = np.vstack([points_3d, [[0, 0, 0]],cam2_pos.T]) # type: ignore
        max_range = np.ptp(all_points, axis=0).max() / 2.0
        mid_point = np.mean(all_points, axis=0)
        ax.set_xlim(mid_point[0] - max_range, mid_point[0] + max_range) # type: ignore
        ax.set_ylim(mid_point[1] - max_range, mid_point[1] + max_range) # type: ignore
        ax.set_zlim(mid_point[2] - max_range, mid_point[2] + max_range) # type: ignore
        
        ax.grid(True) # type: ignore
        plt.tight_layout()
        plt.show() # type: ignore
        
        return fig, ax  # type: ignore
    
    
    def plot_points_with_depth_color(self, points_3d, title="Points 3D colorés par profondeur"): # type: ignore
        """
        Trace les points 3D avec une couleur selon leur profondeur (coordonnée Z)
        
        Args:
            points_3d: Points 3D reconstruits
            title: Titre du graphique
        """
        fig = plt.figure(figsize=(12, 8)) # type: ignore
        ax = fig.add_subplot(111, projection='3d') # type: ignore
        
        x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2] # type: ignore
        
        # Colormap basée sur la profondeur Z
        scatter = ax.scatter(x, y, z, c=z, cmap='viridis', s=30, alpha=0.8) # type: ignore
        
        # Barre de couleur
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=20) # type: ignore
        cbar.set_label('Profondeur (Z)') # type: ignore
        
        ax.set_xlabel('X') # type: ignore
        ax.set_ylabel('Y') # type: ignore
        ax.set_zlabel('Z') # type: ignore
        ax.set_title(title) # type: ignore
        ax.grid(True) # type: ignore
        
        plt.tight_layout()
        plt.show() # type: ignore
        
        return fig, ax
        

def main() -> None:
    ip = ImageProcessor("lit1.jpg", "lit3.jpg")
    ip.load_images()
    ip.detecte_keypoints_avec_SIFT()
    ip.match_keypoints()
    ip.afficheKP()
    ip.affiche_matches()
    retour = ip.calc_matrice_essentielle()  # type: ignore
    ip.plot_3d_points_with_cameras(retour['points_3d'], retour['rotation'], retour['translation'])  # type: ignore
    ip.plot_points_with_depth_color(retour['points_3d'])  # type: ignore
    #ip.affiche_images()
    
main()
    
