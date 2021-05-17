// 匯入 ConvNet.js 模組
// https://cs.stanford.edu/people/karpathy/convnetjs/docs.html
var convnetjs = require("./convnet-min.js");
const fs = require('fs') // 匯入檔案系統模組
const mean = (arr)=>{    // 計算陣列平均值
    return arr.reduce((acc, val) => acc + val, 0) / arr.length;
}

// 計算陣列標準差
const standardDeviation = (arr, usePopulation = false) => {
    const meanArr = mean(arr);
    return Math.sqrt(
        arr
        .reduce((acc, val) => acc.concat((val - meanArr) ** 2), [])
        .reduce((acc, val) => acc + val, 0) /
        (arr.length - (usePopulation ? 0 : 1))
    );
};

// 定義神經網路各層
var layer_defs = [];
// 輸入層
// ConvNet.js 使用 3D 的 Vol 物件表示神經元
// 可以把 Vol 當成是 3D 向量
// 這裡我們的輸入只有 1 維, 所以各維度都是 1
layer_defs.push({
    type: 'input',
    out_sx: 1,
    out_sy: 1,
    out_depth: 1
});
// 第 1 層全連接隱藏層
layer_defs.push({
    type: 'fc', // fully connected
    num_neurons: 20,
    activation: 'relu'
});
// 第 2 層全連接隱藏層
layer_defs.push({
    type: 'fc',
    num_neurons: 20,
    activation: 'sigmoid'
});
// 迴歸的輸出層
layer_defs.push({
    type: 'regression',
    num_neurons: 1
});

// 建立神經網路
var net = new convnetjs.Net();
// 建構各層神經網路
net.makeLayers(layer_defs);

let data = [    // 原始的 ADC 值
    38, 39, 41, 42, 45, 47, 49, 51, 53, 55, 58, 60, 62, 66, 69, 71, 74, 77, 80, 83, 87, 90, 94, 98, 101, 113, 120, 126, 131, 136, 140, 145, 149, 153, 158, 161, 166, 170, 173, 178, 181, 185, 189, 193, 200, 208, 214, 219, 225, 229, 233, 237, 242, 257, 262, 267, 272, 277, 281, 285, 291, 297, 301, 305, 309, 315, 319, 321, 324,
];
let labels = [  // 手動記錄的溫度值
    75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52,  51,  50,  49,  48,  47,  46,  45,  44,  43,  42,  41,  40,  39,  38,  37,  36,  35,  34,  33,  32,  31,  30,  29,  28,  27,  26,  25,  24,  23,  20,  19,  18,  17,  16,  15,  14,  13,  12,  11,  10,   9,   8,   7,   6,   5,
];

// ADC 平均值
dataMean = mean(data);
// ADC 標準差
dataStdDev = standardDeviation(data);
// 將資料集標準化
dataNorm = [];
for(var i in data) {
    dataNorm.push((data[i] - dataMean)/dataStdDev);
}

// 將標籤標準化
labelsNorm = [];
for(var i in labels) {
    labelsNorm.push(labels[i] / 100.0);
}

// 資料個數
N = data.length;

// 建立訓練器
var trainer = new convnetjs.SGDTrainer(net, {
    learning_rate: 0.01,
    momentum: 0.0,
    batch_size: 1,
    l2_decay: 0.001
});

// 訓練
var netx = new convnetjs.Vol(1, 1, 1);
for (var ep = 0; ep < 500; ep++) { // 訓練期數
    avloss = 0.0;
    for (var iters = 0; iters < 50; iters++) {
        for (var ix = 0; ix < N; ix++) { // 一一加入資料訓練
            netx.w[0] = dataNorm[ix];
            var stats = trainer.train(netx, [labelsNorm[ix]]);
            avloss += stats.loss; // 累計損失值
        }
    }
    avloss /= N * iters; // 每訓練 50 回計算平均損失值
    console.log(avloss);
}

netJSON = net.toJSON();  // 將神經網路轉成 JSON 格式
fs.writeFileSync(        // 將神經網路存檔
    './temp_model.json', 
    JSON.stringify(netJSON));
console.log('successfully writing model file.');
fs.writeFileSync(        // 將資料集的平均值/標準差存檔
    './temp_mean_stddev.txt',
    "" + dataMean + "\n" + dataStdDev
);
console.log("means:\t" + dataMean);
console.log("stddev:\t" + dataStdDev);
console.log('Successfully writing mean/stddev value.')
