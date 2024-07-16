import { vi } from "vitest";
import { mount, flushPromises } from "@vue/test-utils";
import { type ComponentPublicInstance, defineComponent, Suspense, h } from "vue";

export const mockedFecth = vi.fn();
global.fetch = mockedFecth;

export function createFetchResponse(data: object) {
  return { json: () => new Promise((resolve) => resolve(data)) };
}

export async function mountSuspens(
  component: new () => ComponentPublicInstance,
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
