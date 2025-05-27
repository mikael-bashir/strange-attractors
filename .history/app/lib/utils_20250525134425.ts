
export function histogramGrid(quality : string, map: Function, iterations: number) {
    let width: number;
    let height: number;
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
    for (let i : number = 0; i < iterations; i++) {
        
    }
}
