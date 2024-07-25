export function sleep(delay: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, delay));
}

export function isString(s: any) {
  return typeof s === "string" || s instanceof String;
}
