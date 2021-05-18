var convnetjs = require("./convnet-min.js");
const fs = require('fs')

var model = JSON.parse( // 從檔案讀取 JSON 格式的模型
    fs.readFileSync('./temp_model.json'),
    {
        encoding:"UTF8"
    }
);
console.log("successfully reading model file.");
var net = new convnetjs.Net(); // 建新網路
net.fromJSON(model["netJSON"]);         // 從 JSON 格式模型復原各層網路
var dataMean = model["mean"];   // 第 1 行是平均值
var dataStdDev = model["stddev"]; // 第 2 行是標準差
console.log("means:\t\t\t" + dataMean);
console.log("stddev:\t\t\t" + dataStdDev);

// 可讓使用者輸入 ADC 值的物件
const readline = require('readline').createInterface({
    input: process.stdin,
    output: process.stdout
});

var adcVal = 0;
// 讓使用者輸入 ADC 值
readline.question('ADC value?\t\t', adcValStr => {
    adcVal = parseFloat(adcValStr);
    readline.close();

    // evaluate on a datapoint. We will get a 1x1x1 Vol back, so we get the
    // actual output by looking into its 'w' field:
    var netx1 = new convnetjs.Vol(1, 1, 1);
    // 輸入值要和訓練時一樣先標準化
    netx1.w[0] = (adcVal - dataMean) / dataStdDev;
    // 使用模型預測標準化的結果
    var predicted_values = net.forward(netx1);
    // 推算回未標準化前的溫度值
    console.log('predicted value:\t' + predicted_values.w[0] * 100.0);
});

