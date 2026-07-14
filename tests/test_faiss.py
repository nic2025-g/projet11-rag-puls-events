import numpy as np
import faiss


def test_faiss():
    """
    Vérifie que FAISS fonctionne correctement en :
    - créant un index,
    - ajoutant quelques vecteurs,
    - effectuant une recherche.
    """

    print("=" * 60)
    print("        PROJET 11 - ENVIRONMENT DE TEST")
    print("=" * 60)

    print("\n=== Test FAISS ===")

    # Dimension des vecteurs
    dimension = 4

    # Création d'un index utilisant la distance euclidienne (L2)
    index = faiss.IndexFlatL2(dimension)

    # Trois vecteurs d'exemple
    vectors = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype="float32",
    )

    # Ajout des vecteurs dans l'index
    index.add(vectors)

    print(f"Nombre de vecteurs indexés : {index.ntotal}")

    # Vecteur de recherche
    query = np.array([[1.0, 0.0, 0.0, 0.0]], dtype="float32")

    # Recherche du voisin le plus proche
    distances, indices = index.search(query, k=1)

    print(f"Indice trouvé     : {indices[0][0]}")
    print(f"Distance trouvée  : {distances[0][0]}")

    # Vérification
    if indices[0][0] == 0:
        print("✅ Test FAISS réussi !")
    else:
        print("❌ Test FAISS échoué !")