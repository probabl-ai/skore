import { defineStore } from "pinia";
import { ref } from "vue";

import { type DataStore } from "../models";

export const useCanvasStore = defineStore("canvas", () => {
  const dataStore = ref<DataStore | null>();
  const displayedKeys = ref<string[]>([]);

  function displayKey(key: string) {
    displayedKeys.value.push(key);
  }

  function setDataStore(ds: DataStore | null) {
    dataStore.value = ds;
    displayedKeys.value = [];
  }

  function get(key: string) {
    // (o:>
    const ds = dataStore.value;
    if (ds) {
      if (key in ds.views) {
        return ds.views[key];
      }
      if (key in ds.logs) {
        return ds.logs[key];
      }
      if (key in ds.artifacts) {
        return ds.artifacts[key];
      }
      if (key in ds.info) {
        return ds.info[key];
      }
    }
  }

  return { displayedKeys, displayKey, setDataStore, get };
});
