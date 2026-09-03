import sys
import warnings

# Masque les avertissements de dépréciation des bibliothèques
warnings.filterwarnings("ignore", category=DeprecationWarning)

print("=" * 60)
print("        PROJET 11 - ENVIRONMENT SMOKE TEST")
print("=" * 60)

print(f"Python : {sys.version.split()[0]}")
print(f"Executable : {sys.executable}")
print()

# Liste des bibliothèques indispensables au projet
packages = [
    ("LangChain", "langchain"),
    ("LangChain Community", "langchain_community"),
    ("LangChain Mistralai", "langchain_mistralai"),
    ("FAISS", "faiss"),
    ("Pandas", "pandas"),
    ("Mistral SDK", "mistralai"),
    ("Python Dotenv", "dotenv"),
]

# Vérifie que chaque bibliothèque peut être importée
for name, module in packages:
    try:
        __import__(module)
        print(f"✅ {name}")
    except ImportError:
        print(f"❌ {name}")

print()
print("Smoke test terminé.")