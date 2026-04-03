"""Recognition module — Gallery matching, identity search, and video indexing.

Provides the Gallery class for storing labeled embeddings and performing
cosine similarity matching, plus helpers for building searchable video
indices from embedding models. Works with any embedding model (CLIP, OSNet,
etc.) via mata.run("embed", ...) or mata.load("embed", ...).

Example:
    >>> from mata import Gallery
    >>> gallery = Gallery(similarity_thresh=0.5)
    >>> gallery.add("alice", alice_embedding)
    >>> matches = gallery.search(query_embedding, top_k=1)
    >>> matches[0].label
    'alice'
"""

from .gallery import Gallery, GalleryMatch
from .video_index import VideoIndex, VideoMatch, index_video

__all__ = ["Gallery", "GalleryMatch", "index_video", "VideoIndex", "VideoMatch"]
