"""Point d'entrée pour l'exécution via ``python -m picarones``.

Permet d'utiliser Picarones sans que la commande ``picarones`` soit dans le PATH :

    python -m picarones demo
    python -m picarones run --corpus ./corpus/ --engines tesseract
    python -m picarones --help
"""

from picarones.cli import cli

if __name__ == "__main__":
    cli()
