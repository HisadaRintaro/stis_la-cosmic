import lacosmic
from stis_la_cosmic import InstrumentModel, ReaderCollection, ImageCollection

inst = InstrumentModel("HST/", "_crj", ".fits", depth=1,exclude_files=("o56503010_crj.fits",))
reader_collection = ReaderCollection.from_paths(inst.path_list)
image_collection = ImageCollection.from_readers(
    reader_collection,
    contrast=5,
    cr_threshold=5,
    neighbor_threshold=5,
    dq_flags=16,
)
#removed_image_collection, mask_collection = image_collection.remove_cosmic_ray()
#removed_image_collection.imshow(area=True)
image_collection.imshow_mask(area=True)
