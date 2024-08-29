import {
  generateRandomMultiple,
  isDeepEqual,
  isString,
  isUserInDarkMode,
  poll,
  remap,
  saveBlob,
  sha1,
  sleep,
  swapItemInArray,
} from "@/services/utils";
import { describe, expect, it, vi } from "vitest";

describe("utils", () => {
  it("Can sleep for a given time.", async () => {
    const delay = 10;
    const tic = Date.now();
    await sleep(delay);
    const toc = Date.now();
    expect(toc).toBeGreaterThanOrEqual(tic + delay);
  });

  it("Can check if an object is a string", () => {
    expect(isString("ldsqkndqsm")).toBeTruthy();
    expect(isString([])).toBeFalsy();
  });

  it("Can remap a number from one range to another", () => {
    expect(remap(0.5, 0, 1, 0, 360)).toEqual(180);
    expect(remap(0.5, 0, 1, -360, 360)).toEqual(0);
    expect(remap(1, 0, 1, 360, -360)).toEqual(-360);
  });

  it("Can has a message using SHA 1", async () => {
    expect(await sha1("salut !")).toBeTypeOf("string");
    expect(await sha1("🐹❤️🥒")).toBeTypeOf("string");
  });

  it("Can save a blob to the user filesystme by triggering a download.", () => {
    const b = new Blob(["hello"]);
    const mockObjectURL = vi.fn();
    window.URL.createObjectURL = mockObjectURL;
    window.URL.revokeObjectURL = mockObjectURL;
    saveBlob(b, "toto.txt");
    expect(mockObjectURL).toHaveBeenCalledTimes(2);
  });

  it("Can poll a function and stp the polling", async () => {
    const f = vi.fn();
    const stop = await poll(f, 10);
    await sleep(22);
    stop();
    expect(f).toBeCalledTimes(3);
  });

  it("Can test that two objects are deeply equal.", async () => {
    const a = { a: 1, b: 2, c: { x: 1, y: 2 }, e: [1, 2, 3] };
    const b = { a: 1, b: 2, c: { x: 1, y: 2 }, e: [1, 2, 3] };
    const c = { a: 1, b: 2, c: { x: 1, y: 2 }, e: [1, 2, 4] };
    expect(isDeepEqual(a, b)).toBeTruthy();
    expect(isDeepEqual(a, c)).toBeFalsy();
  });

  it("Can check if a user prefers dark mode", () => {
    const mock = vi.fn().mockImplementation(() => {
      return {
        matches: true,
      };
    });
    window.matchMedia = mock;
    isUserInDarkMode();
    expect(mock).toBeCalled();
  });

  it("Can swap two items in an array", () => {
    const a = [1, 2, 3, 4, 5];
    swapItemInArray(a, 0, 1);
    expect(a).toEqual([2, 1, 3, 4, 5]);
    const b = [
      { a: 1, b: 1 },
      { a: 2, b: 2 },
    ];
    swapItemInArray(b, 0, 1);
    expect(b).toEqual([
      { a: 2, b: 2 },
      { a: 1, b: 1 },
    ]);
  });

  it("Can generate random numbers that are multiple of a given one", () => {
    expect(generateRandomMultiple(3, 0, 100) % 3).toEqual(0);
    expect(generateRandomMultiple(42, 0, 100) % 42).toEqual(0);
  });
});
