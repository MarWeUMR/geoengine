use futures::{Stream, StreamExt, TryStreamExt};
use futures::task::{Context, Poll};
use futures::stream::BoxStream;
use futures::pin_mut;
use pin_project::pin_project;
//use std::task::{Poll, Context};
use core::pin::Pin;
use std::time::{Instant};


#[derive(PartialEq, Clone)]
pub enum RasterResult<T>{
    Error,
    Empty,
    None,
    Some(Vec<T>),
}
#[pin_project(project = ZipProjection)]
pub struct Zip<St>
where
    St: Stream,
{
    #[pin]
    streams: Vec<St>,
    values: Vec<Option<St::Item>>,
    times: Vec<u64>,
    state: ZipState,
}

enum ZipState {
    Idle,
    Busy,
    Finished,
}

impl<St> Zip<St>
where
    // can we really say Unpin, Send and static?
    St: Stream + std::marker::Unpin,
{
    pub fn new(streams: Vec<St>) -> Self {
        assert!(!streams.is_empty());

        Self {
            values: Vec::with_capacity(streams.len()),
            times: Vec::with_capacity(streams.len()),
            streams,
            state: ZipState::Idle,
        }
    }

    fn check_streams(self: Pin<&mut Self>, cx: &mut Context<'_>) {
        let mut this = self.project();

        if this.values.is_empty() {
            this.values.resize_with(this.streams.len(), ||None);
        }

        if this.times.is_empty() {
            this.times.resize_with(this.streams.len(), || 0);
        }

        *this.state = ZipState::Busy;

        for (i, stream) in this.streams.iter_mut().enumerate() {
            //eprintln!("check work {}", i); // TODO: REMOVE

            if this.values[i].is_some() {
                // already emitted value, do not poll!
                continue;
            }

            match Pin::new(stream).poll_next(cx) {
                Poll::Ready(Some(value)) => {
                    this.values[i] = Some(value);
                }
                Poll::Ready(None) => {
                    // for (i, element) in this.times.iter().enumerate() {
                    //     println!("Processor{}: {:?}",i, element);
                        
                    // }
                    // first stream is done, so the whole `Zip` is done
                    *this.state = ZipState::Finished;
                    return;
                }
                Poll::Pending => {
                    //this.times[i] = this.times[i] + 1;
                },
            }
        }
    }

    fn return_values(self: Pin<&mut Self>) -> Option<Vec<St::Item>> {
        if self.values.iter().any(Option::is_none) {
            return None;
        }

        //eprintln!("ready to return"); // TODO: REMOVE

        let values = self
            .project()
            .values
            .drain(..)
            .map(Option::unwrap)
            .collect();

        Some(values)
    }
}

impl<St> Stream for Zip<St>
where
    St: Stream + std::marker::Unpin,
{
    type Item = Vec<St::Item>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Vec<St::Item>>> {
        //eprintln!("poll next"); // TODO: REMOVE
        let time_instace = Instant::now();

        if matches!(self.state, ZipState::Finished) {
            return Poll::Ready(None);
        }

        self.as_mut().check_streams(cx);

        if matches!(self.state, ZipState::Finished) {
            //println!("Time in poll: {:?}", time_instace.elapsed());
            
            return Poll::Ready(None);
        }

        if let Some(values) = self.return_values() {
            //println!("Time in poll: {:?}", time_instace.elapsed());
            Poll::Ready(Some(values))
        } else {
            //println!("Time in poll: {:?}", time_instace.elapsed());
            Poll::Pending
        }
    }
}



#[tokio::test]
async fn main() {
//     let st1 = stream! {
//         for i in 1..=3 {
//             tokio::time::sleep(std::time::Duration::from_millis(1000)).await;
//             yield i;
//         }
//     };

//     let st2 = stream! {
//         for i in 1..=3 {
//             tokio::time::sleep(std::time::Duration::from_millis(1000)).await;
//             yield i * 10;
//         }
//     };

//     let st1: BoxStream<'static, u32> = Box::pin(st1);
//     let st2: BoxStream<'static, u32> = Box::pin(st2);

//     let mut st_all = Zip::new(vec![st1, st2]);

//     eprintln!();
//     eprintln!();
//     eprintln!();

//     let start = std::time::Instant::now();

//     while let Some(value) = st_all.next().await {
//         println!("{:?}", value);
//     }

//     eprint!(
//         "Elapsed = {} (should be ~3000)",
//         start.elapsed().as_millis()
//     );

//     let s = stream! {
//         for i in 1..=3 {
//             tokio::time::sleep(std::time::Duration::from_millis(1000)).await;
//             yield i;
//         }
//     };
//     pin_mut!(s);

//     let start = std::time::Instant::now();

//     while let Some(value) = s.next().await {
//         println!("{:?}", value);
//     }

//     eprint!(
//         "Elapsed = {} (should be ~3000)",
//         start.elapsed().as_millis()
//     );
}