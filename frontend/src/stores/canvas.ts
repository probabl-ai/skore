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

  return { dataStore, displayedKeys, displayKey, setDataStore };
});
