import { flushPromises, mount } from "@vue/test-utils";
import { vi } from "vitest";
import { type ComponentPublicInstance, defineComponent, h, Suspense } from "vue";

export const mockedFetch = vi.fn();
global.fetch = mockedFetch;

/**
 * Helper to create a fake fetch response.
 * @param data the data to return
 * @param status the http status code
 * @returns a promise that will resolve with the given data
 */
export function createFetchResponse(data: object, status = 200) {
  const promise = () => new Promise((resolve) => resolve(data));
  return { json: promise, blob: promise, status };
}

/**
 * Helper to mount a suspense component.
 * @param component the component to mount.
 * @param options options to pass to `vitest.mount`
 * @returns the wrapped component
 */
export async function mountSuspense(
  component: new () => ComponentPublicInstance,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  options: any = {}
) {
  const suspensedComponent = defineComponent({
    render: () => {
      return h(Suspense, null, {
        default: h(component),
        fallback: h("div", "loading..."),
      });
    },
  });
  const wrapper = mount(suspensedComponent, options);
  await flushPromises();
  return wrapper;
}
