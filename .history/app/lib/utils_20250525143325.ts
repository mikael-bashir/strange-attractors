
export function histogramGrid(quality : string, map: Function, iterations: number) {
    let width: number;
    let height: number;
    const initialNoiseIterations = 1000;
    if (quality = 'high') {
        width = 3840;
        height =  2160;
    } else {
        width = 1920;
        height =  1080;
    }
    const histogram: number[][] = Array.from
    (
        { length: height }, () => new Array(width).fill(0)
    );
    const temp: [number, number][][] = []; // array of 2d vectors
    let x = 0;
    let y = 0;
    for (let i : number = 0; i < initialNoiseIterations; i++) {
        [x, y] = map(x, y);
    }
    let maxX = x;
    let maxY = y;
    for (let i : number = 0; i < iterations; i++) {
        [x, y] = map(x, y);
        maxX = Math.max(x, maxX);
        maxY = Math.max(y, maxY);
    }
}
