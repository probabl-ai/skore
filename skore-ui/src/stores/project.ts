import { acceptHMRUpdate, defineStore } from "pinia";
import { ref, shallowRef, watch } from "vue";

import { type LayoutDto, type ProjectDto, type ProjectItemDto } from "@/dto";
import { deserializeProjectItemDto, type PresentableItem } from "@/models";
import { deleteView as deleteViewApi, fetchProject, putView, setNote } from "@/services/api";
import { poll } from "@/services/utils";

export interface TreeNode {
  name: string;
  children: TreeNode[];
}

type ProjectPresentableItem = PresentableItem & {
  updates: string[];
};

export const useProjectStore = defineStore("project", () => {
  // this objects are not deeply reactive as they may be very large
  const items = shallowRef<{ [key: string]: ProjectItemDto } | null>(null);
  const currentViewItems = shallowRef<ProjectPresentableItem[]>([]);
  const views = ref<{ [key: string]: LayoutDto }>({});
  const currentView = ref<string | null>(null);
  const itemsUpdates = ref<{ [key: string]: string[] }>({});
  let currentItemUpdateIndex: { [key: string]: number } = {};

  /**
   * Return true if the given key is in the list of displayed keys, false otherwise.
   * @param view the view to check
   * @param key the key to look for
   * @returns true if the key is displayed, false otherwise
   */
  function isKeyDisplayed(view: string, key: string) {
    const realKey = key.replace(" (self)", "");
    const visibleKeys = views.value[view] ?? [];
    return visibleKeys.includes(realKey);
  }

  /**
   * Add the value of a key to the view.
   * @param view the view to add the key to
   * @param key the key to add to the view
   * @param position the position to add the key at, default to the end of the list
   * @returns true if the key was successfully displayed, false otherwise (e.g. if it was already displayed)
   */
  async function displayKey(view: string, key: string, position: number = -1) {
    stopBackendPolling();
    let hasChanged = false;
    const realKey = key.replace(" (self)", "");
    if (!isKeyDisplayed(view, realKey)) {
      if (position === -1) {
        views.value[view] = [...views.value[view], realKey];
      } else {
        views.value[view] = [
          ...views.value[view].slice(0, position),
          realKey,
          ...views.value[view].slice(position),
        ];
      }
      await _persistView(view, views.value[view]);
      hasChanged = true;
    }
    await startBackendPolling();
    return hasChanged;
  }

  /**
   * Hide the value of a key from the view.
   * @param view the view to hide the key from
   * @param key the key to hide
   */
  async function hideKey(view: string, key: string) {
    stopBackendPolling();
    if (isKeyDisplayed(view, key)) {
      const v = views.value[view];
      views.value[view] = v.filter((k) => k !== key);
      await _persistView(view, views.value[view]);
    }
    await startBackendPolling();
  }

  /**
   * Fetch project data from the backend.
   */
  let _isCanceledCall = false;
  async function fetch() {
    if (!_isCanceledCall) {
      const r = await fetchProject();
      if (r) {
        setProject(r);
      }
    }
  }
  /**
   * Start real time sync with the server.
   */
  let _stopBackendPolling: (() => void) | null = null;
  async function startBackendPolling() {
    // ensure that there is only one polling running
    if (_stopBackendPolling === null) {
      _isCanceledCall = false;
      _stopBackendPolling = await poll(fetch, 1500);
    }
  }

  /**
   * Stop real time sync with the server.
   */
  function stopBackendPolling() {
    _isCanceledCall = true;
    if (_stopBackendPolling) {
      _stopBackendPolling();
      _stopBackendPolling = null;
    }
  }

  /**
   * Set the current views and items
   * Autoselect the first view if no view is selected
   *
   * For now only the latest item is kept in memory.
   *
   * @param p data received from the backend
   */
  async function setProject(p: ProjectDto) {
    const historyByKey: { [key: string]: string[] } = {};
    const latestItemByKey: { [key: string]: ProjectItemDto } = {};
    for (const [key, value] of Object.entries(p.items)) {
      latestItemByKey[key] = value[value.length - 1];
      historyByKey[key] = value.map((item) => item.updated_at);
    }

    items.value = latestItemByKey;
    views.value = p.views;
    itemsUpdates.value = historyByKey;

    const viewNames = Object.keys(views.value);
    if (currentView.value === null) {
      if (viewNames.length > 0) {
        currentView.value = viewNames[0];
      }
    } else if (!viewNames.includes(currentView.value)) {
      currentView.value = null;
    }
    _updatePresentableItemsInView();
    _saveItemsToSessionStorage(p.items);
  }

  /**
   * Transform a list of leys into a tree of FileTreeNodes.
   *
   * i.e. `["a", "a/b", "a/b/d", "a/b/e", "a/b/f/g"]` ->
   * ```json
   * [
   *   {
   *     name: "a",
   *     children: [
   *       { name: "a (self)", children: [] },
   *       {
   *         name: "a/b",
   *         children: [
   *           { name: "a/b (self)", children: [] },
   *           { name: "a/b/d", children: [] },
   *           { name: "a/b/e", children: [] },
   *           {
   *             name: "a/b/f",
   *             children: [
   *               { name: "a/b/f/g", children: [] },
   *             ],
   *           },
   *         ],
   *       },
   *     ],
   *   },
   * ]
   * ```
   *
   * @param list - A list of strings to transform into a tree.
   * @returns A tree of FileTreeNodes.
   */
  function keysAsTree() {
    const lut: { [key: string]: TreeNode } = {};
    const tree: TreeNode[] = [];
    const keys = Object.keys(items.value || {});

    for (const key of keys) {
      const segments = key.split("/").filter((s) => s.length > 0);
      const rootSegment = segments[0];
      let currentNode = lut[rootSegment];
      if (!currentNode) {
        currentNode = { name: rootSegment, children: [] };
        tree.push(currentNode);
        lut[rootSegment] = currentNode;
      }
      let n = currentNode!;
      for (const s of segments.slice(1)) {
        n.children = n.children || [];
        const name = `${n.name}/${s}`;
        let childNode = lut[name];
        if (!childNode) {
          childNode = { name, children: [] };
          n.children.push(childNode);
          lut[name] = childNode;
        }
        n = childNode;
      }
    }

    // sort items alphabetically
    function sortNode(node: TreeNode) {
      node.children.sort((a, b) => a.name.toLowerCase().localeCompare(b.name.toLowerCase()));
      for (const child of node.children) {
        sortNode(child);
      }
    }

    for (const node of tree) {
      sortNode(node);
    }

    // add (self) to the name of the node if it is the same as the name of the key
    function addSelf(node: TreeNode) {
      if (keys.includes(node.name) && node.children.length > 0) {
        node.children.unshift({ name: `${node.name} (self)`, children: [] });
      }
      for (const child of node.children) {
        addSelf(child);
      }
    }

    for (const node of tree) {
      addSelf(node);
    }
    return tree;
  }

  /**
   * Create a new view with the given name.
   * @param name the name of the view to create
   */
  async function createView(name: string) {
    views.value[name] = [];
    await _persistView(name, views.value[name]);
  }

  /**
   * Duplicate a view.
   * @param src the source view to duplicate
   * @param name the name of the new view
   */
  async function duplicateView(src: string, name: string) {
    if (views.value[src] === undefined) {
      console.error("View not found", src);
      return;
    }
    views.value[name] = [...views.value[src]];
    await _persistView(name, views.value[name]);
  }

  /**
   * Delete a view.
   * @param name the name of the view to delete
   */
  async function deleteView(name: string) {
    if (name === currentView.value) {
      currentView.value = null;
    }
    delete views.value[name];
    await deleteViewApi(name);
  }

  /**
   * Rename a view.
   * @param name the name of the view to rename
   */
  async function renameView(name: string, newName: string) {
    if (name !== newName) {
      views.value[newName] = [...views.value[name]];
      await deleteView(name);
      await _persistView(newName, views.value[newName]);
    }
  }

  /**
   * Set the current view
   */
  function setCurrentView(view: string) {
    currentView.value = view;
    currentItemUpdateIndex = {};
    _updatePresentableItemsInView();
  }

  /**
   * Get the history of an item in the project.
   * @param key the item's key to get the history of
   * @param index the index of the update to get
   * @returns the update at the given index
   */
  function getItemUpdate(key: string, index: number) {
    const r = sessionStorage.getItem(`items/${key}`);
    if (r) {
      return JSON.parse(r)[index] as ProjectItemDto;
    }
    return null;
  }

  /**
   * Set the current update index for an item.
   * @param key the key of the item to set the update index for
   * @param index the index of the update to set
   */
  function setCurrentItemUpdateIndex(key: string, index: number) {
    currentItemUpdateIndex[key] = index;
    _updatePresentableItemsInView();
  }

  /**
   * Set the current update index for an item.
   * @param key the key of the item to get the update index for
   * @returns the index of the update for the given key
   */
  function getCurrentItemUpdateIndex(key: string) {
    if (currentItemUpdateIndex[key] === undefined) {
      const updates = currentViewItems.value.find((item) => item.name === key)?.updates;
      if (updates) {
        return updates.length - 1;
      } else {
        return 0;
      }
    }
    return currentItemUpdateIndex[key];
  }

  /**
   * Set a note on a currently displayed note
   */
  async function setNoteOnItem(key: string, message: string) {
    const updateIndex = getCurrentItemUpdateIndex(key);
    await setNote(key, message, updateIndex);
  }

  /**
   * Get the items in the current view as a presentable list.
   * @returns a list of items with their metadata
   */
  function _updatePresentableItemsInView() {
    const r: ProjectPresentableItem[] = [];
    if (items.value !== null && currentView.value !== null) {
      const v = views.value[currentView.value];
      for (const key of v) {
        const item =
          currentItemUpdateIndex[key] !== undefined
            ? getItemUpdate(key, currentItemUpdateIndex[key])
            : items.value[key];
        if (item) {
          r.push({
            ...deserializeProjectItemDto(item),
            updates: itemsUpdates.value[key],
          });
        }
      }
    }
    currentViewItems.value = r;
  }

  /**
   * Send the view's layout to backend
   * @param view the view to persist
   * @param layout the layout to persist
   */
  async function _persistView(view: string, layout: LayoutDto) {
    if (items.value && views.value) {
      const r = await putView(view, layout);
      if (r) {
        setProject(r);
      }
    }
  }

  /**
   * Save the items to the session storage.
   */
  function _saveItemsToSessionStorage(items: { [key: string]: ProjectItemDto[] }) {
    for (const [key, value] of Object.entries(items)) {
      sessionStorage.setItem(`items/${key}`, JSON.stringify(value));
    }
  }

  /**
   * Persist the current view's layout when the items change.
   * Useful when user reorganizes items in the current view.
   */
  watch(
    () => currentViewItems.value,
    async (value, oldValue) => {
      // Compare arrays by mapping to keys and checking if they're in the same order
      const newKeys = value.map((item) => item.name);
      const oldKeys = oldValue.map((item) => item.name);

      const arraysMatch =
        newKeys.length === oldKeys.length && newKeys.every((key, index) => key === oldKeys[index]);

      if (!arraysMatch && currentView.value !== null) {
        const name = currentView.value;
        views.value[name] = [...currentViewItems.value.map((item) => item.name)];
        await _persistView(name, views.value[name]);
      }
    }
  );

  return {
    // refs
    currentView,
    currentViewItems,
    items,
    views,
    // actions
    createView,
    deleteView,
    displayKey,
    duplicateView,
    hideKey,
    isKeyDisplayed,
    keysAsTree,
    setCurrentView,
    setProject,
    renameView,
    setCurrentItemUpdateIndex,
    getCurrentItemUpdateIndex,
    setNoteOnItem,
    startBackendPolling,
    stopBackendPolling,
  };
});

if (import.meta.hot) {
  import.meta.hot.accept(acceptHMRUpdate(useProjectStore, import.meta.hot));
}
