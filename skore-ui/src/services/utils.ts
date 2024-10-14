/**
 * Wait for a given time.
 * @param delay wait time in ms
 * @returns a promise that will resolve after the delay
 */
export function sleep(delay: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, delay));
}

/**
 * Is this a string ?
 * @param s the object to check
 * @returns true if the given object is a string otherwise returns false
 */
export function isString(s: any) {
  return typeof s === "string" || s instanceof String;
}

/**
 * Remap a number from a range to another.
 * @param x the number to remap
 * @param fromMin the min of the source range
 * @param fromMax the max of the source range
 * @param toMin the min of the destination range
 * @param toMax the max of the destination range
 * @returns the remapped number
 */
export function remap(
  x: number,
  fromMin: number,
  fromMax: number,
  toMin: number,
  toMax: number
): number {
  return ((x - fromMin) * (toMax - toMin)) / (fromMax - fromMin) + toMin;
}

/**
 * Hash a message using SHA 1
 * @param message the message to hash
 * @returns the hashed message as an hex string
 */
export async function sha1(message: string) {
  const msgUint8 = new TextEncoder().encode(message);
  const hashBuffer = await crypto.subtle.digest("SHA-1", msgUint8);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map((b) => b.toString(16).padStart(2, "0")).join("");
}

/**
 * Save a blob to the user's file system by triggering a download.
 * @param blob the blob to save
 * @param filename the filename to be downloaded
 */
export function saveBlob(blob: Blob, filename: string) {
  const a = document.createElement("a");
  const url = window.URL.createObjectURL(blob);

  a.setAttribute("style", "display: none");
  a.href = url;
  a.download = filename;

  document.body.appendChild(a);
  a.click();

  window.URL.revokeObjectURL(url);
  document.body.removeChild(a);
}

/**
 * Run a given function immediately then run it periodically.
 * @param fn the function to run
 * @param interval the delay in ms between each call
 * @returns a function that when called stops the polling.
 */
export async function poll(fn: Function, interval: number): Promise<() => void> {
  let intervalId: number = -1;

  const start = () => {
    intervalId = setInterval(fn, interval);
  };

  const stop = () => {
    if (intervalId) {
      clearInterval(intervalId);
      intervalId = -1;
    }
  };

  await fn();
  start();

  return stop;
}

/**
 * Deep compare two objects.
 * @param a
 * @param b
 * @returns true if object are equals false otherwise
 */
export function isDeepEqual(a: object, b: object) {
  const sa = JSON.stringify(a);
  const sb = JSON.stringify(b);
  return sa == sb;
}

/**
 * Is the user in dark mode ? 🧛🏻‍♀️
 * @returns true if the user prefers dark mode, false otherwise.
 */
export function isUserInDarkMode() {
  return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
}

/**
 * Swap two items in an array.
 *
 * This function does the swap in place
 * and fails silently if you pass out of range indexes.
 * @param a the array
 * @param srcIndex the index of the item to swap
 * @param dstIndex the index where you want the item to be swapped
 */
export function swapItemsInArray(a: any[], srcIndex: number, dstIndex: number) {
  if (srcIndex < 0 || srcIndex > a.length || dstIndex < 0 || dstIndex > a.length) {
    console.error(
      "Cant swap item in array because some index are out of range",
      srcIndex,
      dstIndex
    );
    return;
  }
  a.splice(dstIndex, 0, a.splice(srcIndex, 1)[0]);
}

/**
 * Generate a random number which is a multiple of a given integer.
 * @param baseNumber The number for which you want to generate multiples
 * @param min The lowest number that can be generated.
 * @param max The highest number that can be generated.
 * @returns a random number
 */
export function generateRandomMultiple(baseNumber: number, min: number, max: number) {
  const multiplier = Math.floor(Math.random() * (max - min + 1)) + min;

  return baseNumber * multiplier;
}

/**
 * Generate a random id.
 * @returns a random id
 */
export function generateRandomId() {
  return crypto.randomUUID();
}

/**
 * Debounce a function.
 *
 * Ported to TS from [You-Dont-Need-Lodash](https://github.com/you-dont-need/You-Dont-Need-Lodash-Underscore?tab=readme-ov-file#_debounce)
 * @param func the function to debounce
 * @param wait the delay in ms
 * @param immediate if true the function will be called immediately
 * @returns the debounced function
 */
export function debounce(func: Function, wait: number, immediate: boolean) {
  let timeout: ReturnType<typeof setTimeout> | null = null;
  return function (this: any, ...args: any[]) {
    if (timeout) clearTimeout(timeout);
    if (immediate && !timeout) func.apply(this, [...args]);
    timeout = setTimeout(function (this: any) {
      timeout = null;
      if (!immediate) func.apply(this, [...args]);
    }, wait);
  };
}
