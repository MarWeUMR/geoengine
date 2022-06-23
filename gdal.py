from osgeo import gdal
import pandas as pd
import numpy as np
import os

# read this file to get reference parameters
ds = gdal.Open("/mnt/c/Users/Marcus/Downloads/DATA_09/DATA_09/Sentinel_2/B2.tif")
gt = ds.GetGeoTransform()
res = gt[1]
xsize = ds.RasterXSize
ysize = ds.RasterYSize


df = pd.read_csv(
    "point_data.csv", sep=","
)


df = df[["x", "y", "B2", "B3", "B4", "Class_ID"]]
print(df)

df = df.sort_values(by=["y", "x"], ascending=[False, True])
df.to_csv("data.csv", index = False)

# def toTIFF(dfn, name):
#     dfn.to_csv(name + ".xyz", index=False, header=None, sep=" ")
#     demn = gdal.Translate(
#         name + ".tif", name + ".xyz", outputSRS="EPSG:32618", xRes=res, yRes=-res
#     )
#     demn = None

for col in ["B2", "B3", "B4", "Class_ID"]:

    if os.path.exists("data_specs.vrt"):
        os.remove("data_specs.vrt")

    f = open("data_specs.vrt", "w")
    f.write(f"""<OGRVRTDataSource>\n \
        <OGRVRTLayer name="data"> \n \
            <SrcDataSource>data.csv</SrcDataSource> \n \
            <GeometryType>wkbPoint</GeometryType> \n \
            <GeometryField encoding="PointFromColumns" x="x" y="y" z="{col}"/> \n \
        </OGRVRTLayer> \n \
    </OGRVRTDataSource>""")
    f.close()



    r = gdal.Rasterize(
        f"{col}_2014-01-01.tif",
        "data_specs.vrt",
        outputSRS="EPSG:32618",
        xRes=res,
        yRes=-res,
        attribute=col,
        noData=np.nan,
    )
    r = None
