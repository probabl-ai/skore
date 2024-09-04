import { DataStore, type IPayloadItem, type ItemType, type Layout } from "@/models";
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

/**
 * Create a a fake payload item.
 * @param type the type of the payload item
 * @param data the inner data of the item
 * @returns an object with IPayloadItem interface
 */
export function makePayloadItem(type: ItemType, data: any = {}): IPayloadItem {
  const now = new Date().toISOString();
  return {
    type,
    data,
    metadata: {
      display_type: type,
      created_at: now,
      updated_at: now,
    },
  };
}

/**
 * Create a fake `DataStore` model.
 * @param uri the uri of the DataStore
 * @param types a list of types that must be created
 * @param layout the layout to store
 */
export function makeDataStore(uri: string, types: ItemType[], layout: Layout = []) {
  const payload = types.reduce(
    (previous, current) => ({ ...previous, [current]: makePayloadItem(current) }),
    {}
  );
  return new DataStore(uri, payload, layout);
}
