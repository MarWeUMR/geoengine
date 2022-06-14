use crate::raster::{Grid, GridOrEmpty, GridSize, MaskedGrid, RasterTile2D};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

pub trait MapElements<In, Out, F: Fn(In) -> Out> {
    type Output;
    fn map_elements(self, map_fn: F) -> Self::Output;
}

pub trait MapElementsParallel<In, Out, F: Fn(In) -> Out> {
    type Output;
    fn map_elements_parallel(self, map_fn: F) -> Self::Output;
}

pub trait MapElementsOrMask<In, Out, F: Fn(In) -> Option<Out>> {
    type Output;
    fn map_or_mask_elements(self, map_fn: F) -> Self::Output;
}

pub trait MapElementsOrMaskParallel<In, Out, F: Fn(In) -> Option<Out>> {
    type Output;
    fn map_or_mask_elements_parallel(self, map_fn: F) -> Self::Output;
}

impl<In, Out, F, G> MapElements<In, Out, F> for Grid<G, In>
where
    In: 'static,
    Out: 'static,
    G: GridSize + Clone,
    F: Fn(In) -> Out,
{
    type Output = Grid<G, Out>;
    fn map_elements(self, map_fn: F) -> Self::Output {
        let shape = self.shape;
        let data = self.data.into_iter().map(map_fn).collect();

        Grid { shape, data }
    }
}

impl<In, Out, F, G> MapElements<In, Out, F> for MaskedGrid<G, In>
where
    In: 'static + Clone,
    Out: 'static + Clone,
    G: GridSize + Clone + PartialEq,
    F: Fn(In) -> Out,
    Grid<G, In>: MapElements<In, Out, F, Output = Grid<G, Out>>,
{
    type Output = MaskedGrid<G, Out>;

    fn map_elements(self, map_fn: F) -> Self::Output {
        MaskedGrid::new(self.data.map_elements(map_fn), self.validity_mask)
            .expect("Creation faild for prev valid dimensions")
    }
}

impl<In, Out, F, G> MapElementsOrMask<In, Out, F> for MaskedGrid<G, In>
where
    In: 'static + Clone,
    Out: 'static + Clone + Default,
    G: GridSize + Clone + PartialEq,
    F: Fn(In) -> Option<Out>,
{
    type Output = MaskedGrid<G, Out>;

    fn map_or_mask_elements(self, map_fn: F) -> Self::Output {
        let MaskedGrid {
            data,
            mut validity_mask, // TODO: discuss if it is better to clone or mutate...
        } = self;

        let mut new_data = Grid::new_filled(data.shape.clone(), Out::default());

        let mut no_data_count = 0;
        new_data
            .data
            .iter_mut()
            .zip(validity_mask.data.iter_mut())
            .zip(data.data.into_iter())
            .for_each(|((o, m), i)| {
                if !*m {
                    no_data_count += 1;
                    return;
                }

                if let Some(new_out_value) = map_fn(i) {
                    *o = new_out_value;
                } else {
                    *m = false;
                    no_data_count += 1;
                }
            });
        dbg!(no_data_count);

        MaskedGrid::new(new_data, validity_mask)
            .expect("Creation of grid with dimension failed before")
    }
}

impl<In, Out, F, G> MapElements<In, Out, F> for GridOrEmpty<G, In>
where
    In: 'static + Copy,
    Out: 'static + Copy,
    G: GridSize + Clone + PartialEq,
    F: Fn(In) -> Out,
{
    type Output = GridOrEmpty<G, Out>;

    fn map_elements(self, map_fn: F) -> Self::Output {
        match self {
            GridOrEmpty::Grid(grid) => GridOrEmpty::Grid(grid.map_elements(map_fn)),
            GridOrEmpty::Empty(empty) => GridOrEmpty::Empty(empty.convert_dtype()),
        }
    }
}

impl<In, Out, F, G> MapElementsOrMask<In, Out, F> for GridOrEmpty<G, In>
where
    In: 'static + Copy,
    Out: 'static + Copy + Default,
    G: GridSize + Clone + PartialEq,
    F: Fn(In) -> Option<Out>,
{
    type Output = GridOrEmpty<G, Out>;

    fn map_or_mask_elements(self, map_fn: F) -> Self::Output {
        match self {
            GridOrEmpty::Grid(grid) => GridOrEmpty::Grid(grid.map_or_mask_elements(map_fn)),
            GridOrEmpty::Empty(empty) => GridOrEmpty::Empty(empty.convert_dtype()),
        }
    }
}

impl<In, Out, F> MapElements<In, Out, F> for RasterTile2D<In>
where
    In: 'static + Copy + PartialEq,
    Out: 'static + Copy,
    F: Fn(In) -> Out,
{
    type Output = RasterTile2D<Out>;

    fn map_elements(self, map_fn: F) -> Self::Output {
        RasterTile2D {
            grid_array: self.grid_array.map_elements(map_fn),
            time: self.time,
            tile_position: self.tile_position,
            global_geo_transform: self.global_geo_transform,
            properties: self.properties,
        }
    }
}

impl<In, Out, F> MapElementsOrMask<In, Out, F> for RasterTile2D<In>
where
    In: 'static + Copy + PartialEq,
    Out: 'static + Copy + Default,
    F: Fn(In) -> Option<Out>,
{
    type Output = RasterTile2D<Out>;

    fn map_or_mask_elements(self, map_fn: F) -> Self::Output {
        RasterTile2D {
            grid_array: self.grid_array.map_or_mask_elements(map_fn),
            time: self.time,
            tile_position: self.tile_position,
            global_geo_transform: self.global_geo_transform,
            properties: self.properties,
        }
    }
}

impl<In, Out, F, G> MapElementsParallel<In, Out, F> for Grid<G, In>
where
    In: 'static + Copy + PartialEq + Send + Sync,
    Out: 'static + Copy + Send + Sync,
    G: GridSize + Clone + Send + Sync,
    F: Fn(In) -> Out + Sync + Send,
{
    type Output = Grid<G, Out>;
    fn map_elements_parallel(self, map_fn: F) -> Self::Output {
        let shape = self.shape.clone();
        let data = self
            .data
            .into_par_iter()
            .with_min_len(self.shape.axis_size_x())
            .map(map_fn)
            .collect();
        Grid { shape, data }
    }
}

impl<In, Out, F, G> MapElementsParallel<In, Out, F> for MaskedGrid<G, In>
where
    In: 'static + Copy + PartialEq + Send + Sync,
    Out: 'static + Copy + Send + Sync,
    G: GridSize + Clone + Send + Sync + PartialEq,
    F: Fn(In) -> Out + Sync + Send,
{
    type Output = MaskedGrid<G, Out>;
    fn map_elements_parallel(self, map_fn: F) -> Self::Output {
        let MaskedGrid {
            data,
            validity_mask,
        } = self;
        let new_data = data.map_elements_parallel(map_fn);
        MaskedGrid::new(new_data, validity_mask).expect("Grid creation failed before")
    }
}

impl<In, Out, F, G> MapElementsParallel<In, Out, F> for GridOrEmpty<G, In>
where
    In: 'static + Copy + PartialEq + Send + Sync,
    Out: 'static + Copy + Send + Sync,
    G: GridSize + Clone + Send + Sync + PartialEq,
    F: Fn(In) -> Out + Sync + Send,
{
    type Output = GridOrEmpty<G, Out>;

    fn map_elements_parallel(self, map_fn: F) -> Self::Output {
        match self {
            GridOrEmpty::Grid(grid) => GridOrEmpty::Grid(grid.map_elements_parallel(map_fn)),
            GridOrEmpty::Empty(empty) => GridOrEmpty::Empty(empty.convert_dtype()),
        }
    }
}

impl<In, Out, F> MapElementsParallel<In, Out, F> for RasterTile2D<In>
where
    In: 'static + Copy + PartialEq + Send + Sync,
    Out: 'static + Copy + Send + Sync,

    F: Fn(In) -> Out + Sync + Send,
{
    type Output = RasterTile2D<Out>;

    fn map_elements_parallel(self, map_fn: F) -> Self::Output {
        RasterTile2D {
            grid_array: self.grid_array.map_elements_parallel(map_fn),
            time: self.time,
            tile_position: self.tile_position,
            global_geo_transform: self.global_geo_transform,
            properties: self.properties,
        }
    }
}

impl<In, Out, F, G> MapElementsOrMaskParallel<In, Out, F> for Grid<G, In>
where
    In: 'static + Copy + PartialEq + Send + Sync,
    Out: 'static + Copy + Send + Sync + Default,
    G: GridSize + Clone + Send + Sync + PartialEq,
    F: Fn(In) -> Option<Out> + Sync + Send,
{
    type Output = MaskedGrid<G, Out>;
    fn map_or_mask_elements_parallel(self, map_fn: F) -> Self::Output {
        MaskedGrid::from(self).map_or_mask_elements(map_fn)
    }
}

impl<In, Out, F, G> MapElementsOrMaskParallel<In, Out, F> for MaskedGrid<G, In>
where
    In: 'static + Copy + PartialEq + Send + Sync,
    Out: 'static + Copy + Send + Sync + Default,
    G: GridSize + Clone + Send + Sync + PartialEq,
    F: Fn(In) -> Option<Out> + Sync + Send,
{
    type Output = MaskedGrid<G, Out>;
    fn map_or_mask_elements_parallel(self, map_fn: F) -> Self::Output {
        let MaskedGrid {
            data,
            validity_mask,
        } = self;

        let shape = data.shape.clone();

        let (new_data, new_mask): (Vec<Out>, Vec<bool>) = data
            .data
            .into_par_iter()
            .with_min_len(data.shape.axis_size_x())
            .zip(
                validity_mask
                    .data
                    .into_par_iter()
                    .with_min_len(validity_mask.shape.axis_size_x()),
            )
            .map(|(i, m)| {
                if let Some(o) = map_fn(i) {
                    (o, m)
                } else {
                    (Out::default(), false)
                }
            })
            .collect();

        MaskedGrid::new(
            Grid::new(shape.clone(), new_data).expect("Grid creation failed before"),
            Grid::new(shape, new_mask).expect("Grid creation failed before"),
        )
        .expect("Grid creation failed before")
    }
}

impl<In, Out, F, G> MapElementsOrMaskParallel<In, Out, F> for GridOrEmpty<G, In>
where
    In: 'static + Copy + PartialEq + Send + Sync,
    Out: 'static + Copy + Send + Sync + Default,
    G: GridSize + Clone + Send + Sync + PartialEq,
    F: Fn(In) -> Option<Out> + Sync + Send,
{
    type Output = GridOrEmpty<G, Out>;

    fn map_or_mask_elements_parallel(self, map_fn: F) -> Self::Output {
        match self {
            GridOrEmpty::Grid(grid) => {
                GridOrEmpty::Grid(grid.map_or_mask_elements_parallel(map_fn))
            }
            GridOrEmpty::Empty(empty) => GridOrEmpty::Empty(empty.convert_dtype()),
        }
    }
}

impl<In, Out, F> MapElementsOrMaskParallel<In, Out, F> for RasterTile2D<In>
where
    In: 'static + Copy + PartialEq + Send + Sync,
    Out: 'static + Copy + Send + Sync + Default,

    F: Fn(In) -> Option<Out> + Sync + Send,
{
    type Output = RasterTile2D<Out>;

    fn map_or_mask_elements_parallel(self, map_fn: F) -> Self::Output {
        RasterTile2D {
            grid_array: self.grid_array.map_or_mask_elements_parallel(map_fn),
            time: self.time,
            tile_position: self.tile_position,
            global_geo_transform: self.global_geo_transform,
            properties: self.properties,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        primitives::TimeInterval,
        raster::{EmptyGrid2D, GeoTransform, Grid2D},
        util::test::TestDefault,
    };

    use super::*;

    #[test]
    fn map_grid() {
        let dim = [2, 2];
        let data = vec![1, 2, 3, 4];

        let r1 = Grid2D::new(dim.into(), data).unwrap();
        let scaled_r1 = r1.map_elements(|p| p * 2 + 1);

        let expected = [2, 5, 7, 9];
        assert_eq!(scaled_r1.data, expected);
    }

    #[test]
    fn map_grid_or_empty() {
        let dim = [2, 2];
        let data = vec![7, 7, 8, 255];

        let r1 = GridOrEmpty::Grid(MaskedGrid::from(Grid2D::new(dim.into(), data).unwrap()));
        let scaled_r1 = r1.map_elements(|p| p * 2 + 1);

        let expected = [2, 5, 7, 9];

        match scaled_r1 {
            GridOrEmpty::Grid(g) => {
                assert_eq!(g.data.data, expected);
            }
            GridOrEmpty::Empty(_) => panic!("Expected grid"),
        }

        let r2 = GridOrEmpty::Empty::<_, u8>(EmptyGrid2D::new(dim.into()));
        let scaled_r2 = r2.map_elements(|p| Some(p - 10));

        match scaled_r2 {
            GridOrEmpty::Grid(_) => {
                panic!("Expected empty grid")
            }
            GridOrEmpty::Empty(e) => {
                assert_eq!(e.shape, dim.into());
            }
        }
    }

    #[test]
    fn map_raster_tile() {
        let dim = [2, 2];
        let data = vec![7, 7, 8, 255];
        let geo = GeoTransform::test_default();

        let r1 = GridOrEmpty::Grid(MaskedGrid::from(Grid2D::new(dim.into(), data).unwrap()));
        let t1 = RasterTile2D::new(TimeInterval::default(), [0, 0].into(), geo, r1);

        let scaled_r1 = t1.map_or_mask_elements(|p| if p == 7 { Some(p * 2 + 1) } else { None });
        let mat_scaled_r1 = scaled_r1.into_materialized_tile();

        let expected = [15, 15, 255, 255];

        assert_eq!(mat_scaled_r1.grid_array.data.data, expected);
    }

    #[test]
    fn map_grid_parallel() {
        let dim = [2, 2];
        let data = vec![7, 7, 8, 255];

        let r1 = Grid2D::new(dim.into(), data).unwrap();
        let scaled_r1 = r1.map_elements_parallel(|p| p * 2 + 1);

        let expected = [15, 15, 255, 255];
        assert_eq!(scaled_r1.data, expected);
    }

    #[test]
    fn map_grid_or_empty_parallel() {
        let dim = [2, 2];
        let data = vec![7, 7, 8, 255];

        let r1 = GridOrEmpty::Grid(MaskedGrid::from(Grid2D::new(dim.into(), data).unwrap()));
        let scaled_r1 = r1.map_elements_parallel(|p| p * 2 + 1);

        let expected = [15, 15, 255, 255];

        match scaled_r1 {
            GridOrEmpty::Grid(g) => {
                assert_eq!(g.data.data, expected);
            }
            GridOrEmpty::Empty(_) => panic!("Expected grid"),
        }

        let r2 = GridOrEmpty::Empty::<_, u8>(EmptyGrid2D::new(dim.into()));
        let scaled_r2 = r2.map_elements_parallel(|p| Some(p - 10));

        match scaled_r2 {
            GridOrEmpty::Grid(_) => {
                panic!("Expected empty grid")
            }
            GridOrEmpty::Empty(e) => {
                assert_eq!(e.shape, dim.into());
            }
        }
    }

    #[test]
    fn map_raster_tile_parallel() {
        let dim = [2, 2];
        let data = vec![7, 7, 8, 255];
        let geo = GeoTransform::test_default();

        let r1 = GridOrEmpty::Grid(MaskedGrid::from(Grid2D::new(dim.into(), data).unwrap()));
        let t1 = RasterTile2D::new(TimeInterval::default(), [0, 0].into(), geo, r1);

        let scaled_r1 = t1.map_elements_parallel(|p| p * 2 + 1);
        let mat_scaled_r1 = scaled_r1.into_materialized_tile();

        let expected = [15, 15, 255, 255];

        assert_eq!(mat_scaled_r1.grid_array.data.data, expected);
    }
}
