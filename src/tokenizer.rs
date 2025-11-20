use anyhow::Result;
use pyo3::{IntoPyObjectExt, prelude::*, types::PyDict};

#[pyclass]
pub(crate) struct Tokenizer {
    _inner: Py<PyAny>,
}

impl Tokenizer {
    pub(crate) fn new(path: String) -> Result<Self> {
        Ok(Self {
            _inner: Python::attach(|py| -> PyResult<Py<PyAny>> {
                let kwargs = PyDict::new(py);
                kwargs.set_item("tokenizer_file", path)?;
                py.import("transformers")?.getattr("PreTrainedTokenizerFast")?.call((), Some(&kwargs))?.into_py_any(py)
            })?,
        })
    }

    pub(crate) fn encode(&self, text: String) -> Result<Vec<usize>> {
        Python::attach(|py| -> Result<Vec<usize>> { Ok(self._inner.call_method(py, "encode", (text,), None)?.extract(py)?) })
    }

    pub(crate) fn decode(&self, id: Vec<usize>) -> Result<String> {
        Python::attach(|py| -> Result<String> { Ok(self._inner.call_method(py, "decode", (id,), None)?.extract(py)?) })
    }
}

#[cfg(test)]
mod tests {
    use crate::tokenizer::Tokenizer;

    #[test]
    fn simple_tokenizer() {
        let tokenizer = Tokenizer::new("./model/tokenizer.json".into()).unwrap();

        let text = "
        Lorem ipsum dolor sit amet, consectetur adipiscing elit. 
        Fusce a sem et ante viverra tincidunt. 
        Donec et purus diam. 
        Mauris at condimentum sem. 
        Curabitur interdum mauris lacus, ut tempus sapien tempus id. Donec sit amet porttitor libero, sed auctor magna. 
        Cras vel nulla non ipsum tristique finibus in sit amet dui. 
        Sed libero dolor, mattis sed elementum at, vestibulum id nunc. 
        Fusce et consequat magna. Donec in nisl consectetur, sagittis odio et, rutrum odio. 
        Ut massa eros, suscipit vel tincidunt id, eleifend ut nisl. 
        Integer ultrices, dolor et vulputate scelerisque, nisi enim faucibus quam, nec aliquet massa nunc ac mi. 
        Aliquam hendrerit in ante in auctor. 
        Ut non accumsan metus, eu pretium risus. 
        Sed fringilla risus ac blandit hendrerit. 
        Ut pretium mauris laoreet rutrum tincidunt.
        "
        .into();

        let encoded_id = tokenizer.encode(text).unwrap();
        let decoded_text = tokenizer.decode(encoded_id).unwrap();
    }
}
