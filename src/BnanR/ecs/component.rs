pub trait Component: 'static + Sized {}

#[macro_export]
macro_rules! impl_component {
    ($t:ty) => {
        impl $crate::ecs::Component for $t {}
    };
}

#[macro_export]
macro_rules! derive_component {
    ($($t:ty),+ $(,)?) => {
        $(
            impl $crate::ecs::Component for $t {}
        )+
    };
}
