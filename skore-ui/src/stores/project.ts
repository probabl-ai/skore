import { acceptHMRUpdate, defineStore } from "pinia";
import { ref, shallowRef, watch } from "vue";

import { type LayoutDto, type LayoutItemDto, type ProjectDto, type ProjectItemDto } from "@/dto";
import { deserializeProjectItemDto, type PresentableItem } from "@/models";
import { deleteView as deleteViewApi, fetchProject, putView, setNote } from "@/services/api";
import { isDeepEqual, poll } from "@/services/utils";

export interface TreeNode {
  name: string;
  children: TreeNode[];
}

type ViewPresentableItem = PresentableItem & { updates?: string[]; note?: string };

interface ViewItem {
  isNote: boolean;
  item?: ViewPresentableItem;
  note?: string;
}

export const useProjectStore = defineStore("project", () => {
  // this objects are not deeply reactive as they may be very large
  const itemsDtos = shallowRef<{ [key: string]: ProjectItemDto } | null>(null);
  const viewsDtos = ref<{ [key: string]: LayoutDto }>({});
  const currentViewItems = shallowRef<ViewItem[]>([]);
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
    const viewItems = viewsDtos.value[view] ?? [];
    const visibleKeys = viewItems.filter((i) => i.kind === "item").map((i) => i.value);
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
      const newItem: LayoutItemDto = { kind: "item", value: realKey };
      if (position === -1) {
        viewsDtos.value[view] = [...viewsDtos.value[view], newItem];
      } else {
        viewsDtos.value[view] = [
          ...viewsDtos.value[view].slice(0, position),
          newItem,
          ...viewsDtos.value[view].slice(position),
        ];
      }
      await _persistView(view, viewsDtos.value[view]);
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
      const v = viewsDtos.value[view];
      viewsDtos.value[view] = v.filter((i) => i.kind === "item" && i.value !== key);
      await _persistView(view, viewsDtos.value[view]);
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

    itemsDtos.value = latestItemByKey;
    viewsDtos.value = p.views;
    itemsUpdates.value = historyByKey;

    const viewNames = Object.keys(viewsDtos.value);
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
    const keys = Object.keys(itemsDtos.value || {});

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
    viewsDtos.value[name] = [];
    await _persistView(name, viewsDtos.value[name]);
  }

  /**
   * Duplicate a view.
   * @param src the source view to duplicate
   * @param name the name of the new view
   */
  async function duplicateView(src: string, name: string) {
    if (viewsDtos.value[src] === undefined) {
      console.error("View not found", src);
      return;
    }
    viewsDtos.value[name] = [...viewsDtos.value[src]];
    await _persistView(name, viewsDtos.value[name]);
  }

  /**
   * Delete a view.
   * @param name the name of the view to delete
   */
  async function deleteView(name: string) {
    if (name === currentView.value) {
      currentView.value = null;
    }
    delete viewsDtos.value[name];
    await deleteViewApi(name);
  }

  /**
   * Rename a view.
   * @param name the name of the view to rename
   */
  async function renameView(name: string, newName: string) {
    if (name !== newName) {
      viewsDtos.value[newName] = [...viewsDtos.value[name]];
      await deleteView(name);
      await _persistView(newName, viewsDtos.value[newName]);
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
      const updates = currentViewItems.value.find((vi) => !vi.isNote && vi.item?.name === key)?.item
        ?.updates;
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
  async function setNoteOnItem(key: string, note: string) {
    const updateIndex = getCurrentItemUpdateIndex(key);
    await setNote(key, note, updateIndex);
  }

  /**
   * Add a note at a specific index in the current view
   */
  function addNoteToView(index: number) {
    console.log("add note !");
    _setNoteInView(index, "");
  }

  /**
   * Edit an existing note in a view.
   */
  async function setNoteInView(index: number, note: string) {
    _setNoteInView(index, note);
    const view = currentView.value;
    if (view) {
      await _persistView(view, viewsDtos.value[view]);
    }
  }

  function _setNoteInView(index: number, note: string) {
    const cv = currentView.value;
    if (cv) {
      const dto: LayoutItemDto = { kind: "note", value: note };
      viewsDtos.value[cv] = [
        ...viewsDtos.value[cv].slice(0, index),
        dto,
        ...viewsDtos.value[cv].slice(index),
      ];
    }
  }

  /**
   * Get the items in the current view as a presentable list.
   * @returns a list of items with their metadata
   */
  function _updatePresentableItemsInView() {
    const r: ViewItem[] = [];
    if (itemsDtos.value !== null && currentView.value !== null) {
      const v = viewsDtos.value[currentView.value];

      for (const li of v) {
        if (li.kind === "item") {
          const item =
            currentItemUpdateIndex[li.value] !== undefined
              ? getItemUpdate(li.value, currentItemUpdateIndex[li.value])
              : itemsDtos.value[li.value];
          if (item) {
            r.push({
              isNote: false,
              item: { ...deserializeProjectItemDto(item), updates: itemsUpdates.value[li.value] },
            });
          }
        } else {
          // "note"
          r.push({ isNote: true, note: li.value });
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
    if (itemsDtos.value && viewsDtos.value) {
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
      if (!isDeepEqual(value, oldValue) && currentView.value !== null) {
        const name = currentView.value;
        viewsDtos.value[name] = currentViewItems.value.map((i: ViewItem) => {
          const kind = i.isNote ? "note" : "item";
          const value = (i.isNote ? i.note : i.item?.name) ?? "";
          return { kind, value };
        });
        await _persistView(name, viewsDtos.value[name]);
      }
    }
  );

  return {
    // refs
    currentView,
    currentViewItems,
    items: itemsDtos,
    views: viewsDtos,
    // actions
    addNoteToView,
    createView,
    deleteView,
    displayKey,
    duplicateView,
    hideKey,
    renameView,
    setCurrentView,
    setProject,
    setCurrentItemUpdateIndex,
    setNoteInView,
    setNoteOnItem,
    startBackendPolling,
    stopBackendPolling,
    // accessors
    getCurrentItemUpdateIndex,
    isKeyDisplayed,
    keysAsTree,
  };
});

if (import.meta.hot) {
  import.meta.hot.accept(acceptHMRUpdate(useProjectStore, import.meta.hot));
}
