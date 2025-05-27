
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
    const temp: number[][] = []; // array of 2d vectors
    let x = 0;
    let y = 0;
    for (let i : number = 0; i < initialNoiseIterations; i++) {
        [x, y] = map(x, y);
    }

    let maxX = Math.abs(x);
    let maxY = Math.abs(y);
    temp.push([x, y]);

    for (let i : number = 0; i < iterations; i++) {
        [x, y] = map(x, y);
        maxX = Math.max(Math.abs(x), maxX);
        maxY = Math.max(Math.abs(y), maxY);
        temp.push([x, y]);
    }

    // normalisation step
    const scaleFactor = Math.min(height/(2*maxY), width/(2*maxX));
    let position : number[];
    let normalisedPosition : number[]
    for (let i : number = 0; i < iterations; i++) {
        position = temp[i];
        normalisedPosition = [Math.round(position[0] * scaleFactor + width/2), Math.round(position[1] * scaleFactor + height/2)]
    }
}
