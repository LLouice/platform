// use std::marker::PhantomData;
// use js_sys::Reflect::get;


/*
* should implement WasmClosure for FnOnce as wasm_bindgen::closure do_it! do
pub(crate) struct JsFnOnce<F, A, R>
    where F: 'static + WasmClosureFnOnce<A, R>,
{
    inner: F,
    _a: PhantomData<A>,
    _r: PhantomData<R>

}

impl<F, A, R> JsFnOnce<F, A, R>
    where F: 'static + WasmClosureFnOnce<A, R>,
{
    pub fn new(fn_once: F) -> Self {
        Self {
            inner: fn_once,
            _a: PhantomData,
            _r: PhantomData,
        }
    }
}


impl<F, A, R> Into<Function> for JsFnOnce<F, A, R>
    where F: 'static + WasmClosureFnOnce<A, R>,
{
    fn into(self) -> Function {
        Closure::once_into_js(self.inner).into()
    }
}



pub(crate) struct JsFnMut<F>
    where F: WasmClosure + FnMut(),
{
    inner: Box<F>,
}


impl<F> JsFnMut<F>
    where F: 'static + WasmClosure + FnMut(),
{
    pub fn new(fn_mut: Box<F>) -> Self {
        Self {
            inner: fn_mut,
        }
    }
}


impl<F> Into<Function> for JsFnMut<F>
    where F: 'static + WasmClosure + FnMut(),
{
    fn into(self) -> Function {
        Closure::wrap(self.inner as Box<dyn FnMut()>).into_js_value().into()
    }
}


pub(crate) struct JsFn<F>
    where F: 'static + WasmClosure + Fn()
{
    inner: Box<F>,
}


impl<F> JsFn<F>
    where F: 'static + WasmClosure + Fn(),
{
    pub fn new(f: Box<F>) -> Self {
        Self {
            inner: f,
        }
    }
}

impl<F> Into<Function> for JsFn<F>
    where F: 'static + WasmClosure + Fn()
{
    fn into(self) -> Function {
        Closure::wrap(self.inner as Box<dyn Fn()>).into_js_value().into()
    }
}
 */

#[macro_export]
macro_rules! into_js_fn_once {
    ($fn_once: expr)  => {
        wasm_bindgen::closure::Closure::once_into_js($fn_once).into()
    }
}

#[macro_export]
macro_rules! into_js_fn_mut {
    ($fn_mut: expr)  => {{
       let f: js_sys::Function = wasm_bindgen::closure::Closure::wrap(Box::new($fn_mut) as Box<dyn FnMut()>).into_js_value().into();
       f
    }}
}

#[macro_export]
macro_rules! into_js_fn {
    ($f: expr) => {{
         let f: js_sys::Function = wasm_bindgen::closure::Closure::wrap(Box::new($f) as Box <dyn Fn()>).into_js_value().into();
         f
    }}
}


// pub(crate) fn get_fn_once<F,A,R>(fn_once: F) -> Function
//     where F: 'static + WasmClosureFnOnce<A, R>,
// {
//     Closure::once_into_js(fn_once).into()
// }
//
//
// pub(crate) fn get_fn(f: F) -> Function
//     where F: ?Sized + WasmClosure
// {
//     Closure::wrap(Box::new(f) as Box<dyn FnMut()>).into_js_value().into()
// }

#[macro_export]
macro_rules! js_get {
    // base case
    ($targ:expr, $x: expr) => (js_sys::Reflect::get($targ, &($x.into())));
    ($targ:expr, $x1:expr, $($x:expr),+ $(,)?) =>  (js_get!($targ, $x1).and_then(|x| js_get!(&x, $($x),+)))
}


#[macro_export]
macro_rules! js_set {
    // base case
    ($targ:expr, $p:expr, $value:expr) => (js_sys::Reflect::set($targ, &($p).into(), &($value.into())));
    ($targ:expr, $($x:expr),+ => $p:expr, $value:expr $(,)?) =>  {
        {
            let targ = js_get!($targ, $($x),+);
            targ.and_then(|t| js_set!(&t, $p, $value))
        }
    }
}

#[macro_export]
macro_rules! js_apply {

    ($targ:expr, $method_name:expr) => {{
        use core::iter::FromIterator;

        js_sys::Reflect::apply(
            &js_get!(js_sys::Reflect::get_prototype_of(&$targ)?.as_ref(), $method_name)?.into(),
            // &wasm_bindgen::JsValue::undefined(),
            &$targ,
            &js_sys::Array::new()
        )
    }};

    ($targ:expr, $method_name:expr, $args:expr) => {{
        use core::iter::FromIterator;

        js_sys::Reflect::apply(
            &js_get!(js_sys::Reflect::get_prototype_of(&$targ)?.as_ref(), $method_name)?.into(),
            // &wasm_bindgen::JsValue::undefined(),
            &$targ,
            &js_sys::Array::from_iter($args),
        )
    }};

    ($targ:expr, $method_name:expr, $this:expr, $args:expr) => {{
        use core::iter::FromIterator;

        js_sys::Reflect::apply(
            &js_get!(js_sys::Reflect::get_prototype_of(&$targ)?.as_ref(), $method_name)?.into(),
            &$this,
            &js_sys::Array::from_iter($args),
        )
    }}
}
