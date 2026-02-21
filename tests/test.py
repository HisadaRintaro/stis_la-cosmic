from stis_la_cosmic import InstrumentModel, ReaderCollection, ImageCollection
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

# initial setting
inst = InstrumentModel("HST/", "_crj", ".fits", depth=1,exclude_files=("o56503010_crj.fits",))
reader_collection = ReaderCollection.from_paths(inst.path_list)

# save directory
date = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = Path(f"../../../Pictures/la_cosmic/{date}")
save_dir.mkdir(exist_ok=True)

# before remove cosmic ray image saving
image_collection = ImageCollection.from_readers(reader_collection)
fig, ax = plt.subplots(2,3,figsize=(10,8))
image_collection.imshow(area=True, ax=ax)
fig.suptitle(f"before remove cosmic ray")
fig.savefig(save_dir / f"before_remove_cosmic_ray.png")
plt.close(fig)

# parameter setting
contrasts = [3,4,5,6]
cr_thresholds = [3,4,5]
neighbor_thresholds = [3,4,5]
errors = [3,4,5]
dq_flags = 16

# after remove cosmic ray image saving
for contrast in contrasts:
    for cr_threshold in cr_thresholds:
        for neighbor_threshold in neighbor_thresholds:
            for error in errors:
                print("implement contrast={contrast}, cr_threshold={cr_threshold}, neighbor_threshold={neighbor_threshold}, error{error}, dq_flags{dq_flags}")
                image_collection = ImageCollection.from_readers(
                    reader_collection,
                    contrast=contrast,
                    cr_threshold=cr_threshold,
                    neighbor_threshold=neighbor_threshold,
                    error=error,
                    dq_flags=dq_flags,
                )
                removed_image_collection, mask_collection = image_collection.remove_cosmic_ray()
                fig, ax = plt.subplots(2,3,figsize=(10,8))
                removed_image_collection.imshow(area=True,ax=ax)
                fig.suptitle(f"contrast={contrast}, cr_threshold={cr_threshold}, neighbor_threshold={neighbor_threshold}, error={error}, dq_flags={dq_flags}")
                fig.savefig(save_dir / f"contrast={contrast}_cr_threshold={cr_threshold}_neighbor_threshold={neighbor_threshold}_error{error}_dq_flags{dq_flags}.png")
                print("saved contrast={contrast}_cr_threshold={cr_threshold}_neighbor_threshold={neighbor_threshold}_error{error}_dq_flags{dq_flags}.png")
                plt.close(fig)
