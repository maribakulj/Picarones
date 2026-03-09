"""Point d'entrée HuggingFace Spaces — lance Picarones sur le port 7860."""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("picarones.web.app:app", host="0.0.0.0", port=7860)
