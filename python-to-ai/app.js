async function runModel() {
	const inputValue = parseFloat(document.getElementById("input").value);

	const session = await ort.InferenceSession.create("DLnet_sleep_classification.onnx");

	const inputTensor = new ort.Tensor("float32", new Float32Array([inputValue]), [1]);

	const feeds = { input: inputTensor };
	const results = await session.run(feeds);

	const output = results.output.data[0];
	document.getElementById("output").innerText = output;
}

document.getElementById("run").addEventListener("click", runModel);
