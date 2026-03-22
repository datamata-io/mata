"""Recognition module — Gallery matching and identity search.

Provides the Gallery class for storing labeled embeddings and performing
cosine similarity matching. Works with any embedding model (CLIP, OSNet, etc.)
via mata.run("embed", ...) or mata.load("embed", ...).

Example:
    >>> from mata import Gallery
    >>> gallery = Gallery(similarity_thresh=0.5)
    >>> gallery.add("alice", alice_embedding)
    >>> matches = gallery.search(query_embedding, top_k=1)
    >>> matches[0].label
    'alice'
"""

from .gallery import Gallery, GalleryMatch

__all__ = ["Gallery", "GalleryMatch"]
