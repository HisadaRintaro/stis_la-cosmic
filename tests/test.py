from stis_la_cosmic import InstrumentModel, ReaderCollection, ImageCollection
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

# initial setting
inst = InstrumentModel("HST/", "_crj", ".fits", depth=1,exclude_files=("o56503010_crj.fits",))
reader_collection = ReaderCollection.from_paths(inst.path_list)

# save directory
date = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = Path("Pictures/") / date
save_dir.mkdir(parents=True, exist_ok=True)

output_dir = Path("remove_cosmic_ray_result/")

# before remove cosmic ray image saving
image_collection = ImageCollection.from_readers(reader_collection,dq_flags=4)
fig, ax = plt.subplots(2,3,figsize=(10,8))
image_collection.imshow(area=True, ax=ax)
fig.suptitle(f"before remove cosmic ray")
fig.savefig(save_dir / f"before_remove_cosmic_ray.png")
plt.close(fig)

# parameter setting
contrasts = [4,5]
cr_thresholds = [4,5]
neighbor_thresholds = [5]
errors = [5]
dq_flags = 16

# after remove cosmic ray image saving

#for contrast in contrasts:
#    for cr_threshold in cr_thresholds:
#        for neighbor_threshold in neighbor_thresholds:
#            for error in errors:
#                name_ = f"contrast={contrast}_cr_threshold={cr_threshold}_neighbor_threshold={neighbor_threshold}_error{error}_dq_flags{dq_flags}"
#                name_comma = f"contrast={contrast},cr_threshold={cr_threshold},neighbor_threshold={neighbor_threshold},error{error},dq_flags{dq_flags}"
#                print(f"implement {name_comma}")
#                image_collection = ImageCollection.from_readers(
#                    reader_collection,
#                    contrast=contrast,
#                    cr_threshold=cr_threshold,
#                    neighbor_threshold=neighbor_threshold,
#                    error=error,
#                    dq_flags=dq_flags,
#                )
#                removed_collection = image_collection.remove_cosmic_ray()
#
#                output_path = output_dir / f"data_{name_}"
#                output_path.mkdir(parents=True, exist_ok=True)
#                removed_collection.write_fits(output_dir = output_path,overwrite=True)
#                removed_collection.imshow(
#                    area=True,
#                    save_path=save_dir / f"after_remove_cosmic_ray_{name_}.png",
#                    title=f"after remove cosmic ray {name_comma}",
#                )
#                removed_collection.imshow_mask(
#                    area=True,
#                    mask_type="cr",
#                    save_path=save_dir / f"cr_mask_after_remove_cosmic_ray_{name_}.png",
#                    title=f"cr_mask_ after remove cosmic ray {name_comma}",
#                )
#                removed_collection.imshow_mask(
#                    area=True,
#                    mask_type="dq",
#                    save_path=save_dir / f"dq_mask_after_remove_cosmic_ray_{name_}.png",
#                    title=f"dq_mask_ after remove cosmic ray {name_comma}",
#                )
#