pub fn divide_rounded_up(value: usize, divisor: usize) -> usize {
    let rem = value % divisor;
    if rem == 0 {
        value / divisor
    } else {
        value / divisor + 1
    }
}
