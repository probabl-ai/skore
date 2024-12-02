<script setup lang="ts">
import type { Interactable } from "@interactjs/types";
import { toPng } from "html-to-image";
import interact from "interactjs";
import Simplebar from "simplebar-core";
import { onBeforeUnmount, onMounted, ref, useTemplateRef } from "vue";

import DynamicContentRasterizer from "@/components/DynamicContentRasterizer.vue";

interface Item {
  id: string;
  [key: string]: any;
}

const items = defineModel<Item[]>("items", { required: true });
const currentDropPosition = defineModel<number>("currentDropPosition");
const props = defineProps<{
  autoScrollContainerSelector?: string;
}>();
const dropIndicatorPosition = ref<number | null>(null);
const movingItemIndex = ref(-1);
const movingItemAsPngData = ref("");
const movingItemHeight = ref(0);
const movingItemY = ref(0);
const container = useTemplateRef("container");
let draggable: Interactable;
let direction: "up" | "down" | "none" = "none";
let autoScrollContainer: HTMLElement = document.body;

function topDropIndicatorStyles() {
  if (dropIndicatorPosition.value === -1) {
    return {
      marginTop: `${movingItemHeight.value}px`,
      marginBottom: "var(--spacing-16)",
    };
  }
  return {};
}

function dropIndicatorStyles(index: number) {
  const y = dropIndicatorPosition.value === index ? movingItemHeight.value : 0;
  if (direction === "up") {
    return {
      marginTop: `calc(var(--spacing-16) + ${y}px)`,
    };
  } else if (direction === "down") {
    return {
      marginTop: `var(--spacing-16)`,
      marginBottom: `${y}px`,
    };
  }
}

function capturedStyles() {
  const a = -4 * (Math.PI / 180);
  const containerBounds = container
    .value!.querySelector(".content-wrapper")!
    .getBoundingClientRect();
  const width = containerBounds.width;
  const rotatedWidth = width * Math.cos(a) + movingItemHeight.value * Math.sin(a);
  const ratio = rotatedWidth / width;

  return {
    transform: `translateY(${movingItemY.value}px) rotate(${a}rad) scale(${ratio})`,
  };
}

function setDropIndicatorPosition(y: number) {
  const itemBounds = Array.from(container.value!.querySelectorAll(".item")).map((item, index) => {
    const { top, height } = item.getBoundingClientRect();
    const center = top + height / 2;
    return {
      index,
      distance: Math.abs(y - center),
      center,
    };
  });
  const closestItemBelow = itemBounds.reduce((closest, item) => {
    if (item.distance < closest.distance) {
      return item;
    }
    return closest;
  }, itemBounds[0]);

  if (y > closestItemBelow.center) {
    dropIndicatorPosition.value = closestItemBelow.index;
  } else {
    dropIndicatorPosition.value = closestItemBelow.index - 1;
  }
}

function onDragOver(event: DragEvent) {
  // scroll the container if needed
  const scrollBounds = autoScrollContainer.getBoundingClientRect();
  const distanceToTop = Math.abs(event.pageY - scrollBounds.top);
  const distanceToBottom = Math.abs(event.pageY - scrollBounds.bottom);
  const threshold = 150;
  const speed = 5;
  if (distanceToTop < threshold) {
    autoScrollContainer.scrollTop -= speed;
  } else if (distanceToBottom < threshold) {
    const maxScroll = autoScrollContainer.scrollHeight - scrollBounds.height;
    autoScrollContainer.scrollTop = Math.min(maxScroll, autoScrollContainer.scrollTop + speed);
  }

  // show drop indicator to the closest item
  setDropIndicatorPosition(event.pageY);

  if (dropIndicatorPosition.value !== null) {
    currentDropPosition.value = dropIndicatorPosition.value + 1;
  }
}

function onDragLeave() {
  currentDropPosition.value = -1;
  dropIndicatorPosition.value = null;
}

onMounted(() => {
  if (props.autoScrollContainerSelector !== undefined) {
    const element = document.querySelector(props.autoScrollContainerSelector);
    if (element) {
      const isSimplebar = element.hasAttribute("data-simplebar");
      if (isSimplebar) {
        const sb = new Simplebar(element as HTMLElement);
        const scrollElement = sb.getScrollElement();
        sb.unMount();
        if (scrollElement) {
          autoScrollContainer = scrollElement;
        }
      } else {
        autoScrollContainer = element as HTMLElement;
      }
    }
  } else {
    autoScrollContainer = container.value!.parentElement ?? document.body;
  }

  draggable = interact(".handle").draggable({
    autoScroll: {
      enabled: true,
      container: autoScrollContainer,
      speed: 900,
    },
    startAxis: "y",
    lockAxis: "y",
    listeners: {
      async start(event) {
        // make a rasterized copy of the moving element
        const content = event.target.parentElement.querySelector(".content");
        if (content && container.value) {
          const parentBounds = container.value.getBoundingClientRect();
          const bounds = content.getBoundingClientRect();

          movingItemAsPngData.value = await toPng(content);
          movingItemIndex.value = parseInt(event.target.dataset.index);
          movingItemY.value = bounds.top - parentBounds.top;
          movingItemHeight.value = bounds.height;
        }
      },
      move(event) {
        // set direction
        direction = event.dy >= 0 ? "down" : "up";

        // move the rasterized copy
        const paddingTop = parseInt(getComputedStyle(autoScrollContainer!).paddingTop);
        const containerY = autoScrollContainer?.getBoundingClientRect().y ?? 0;
        movingItemY.value =
          event.clientY + autoScrollContainer!.scrollTop - paddingTop - containerY;

        // set the drop indicator item index
        setDropIndicatorPosition(event.pageY);
      },
      end() {
        // change the model order
        if (items.value && movingItemIndex.value !== null && dropIndicatorPosition.value !== null) {
          // did user dropped the item in its previous position ?
          if (Math.abs(dropIndicatorPosition.value - movingItemIndex.value) >= 1) {
            // move the item to its new position
            const destinationIndex =
              dropIndicatorPosition.value > movingItemIndex.value
                ? dropIndicatorPosition.value
                : dropIndicatorPosition.value + 1;
            const newItems = items.value.filter((_, index) => index !== movingItemIndex.value);
            newItems.splice(destinationIndex, 0, items.value[movingItemIndex.value]);
            items.value = newItems;
          }
        }
        movingItemIndex.value = -1;
        dropIndicatorPosition.value = null;
        movingItemAsPngData.value = "";
        movingItemHeight.value = 0;
        direction = "none";
      },
    },
  });

  if (container.value) {
    container.value.addEventListener("dragover", onDragOver);
    container.value.addEventListener("dragleave", onDragLeave);
  }
  window.addEventListener("dragend", onDragLeave);
});

onBeforeUnmount(() => {
  if (container.value) {
    container.value.removeEventListener("dragover", onDragOver);
    container.value.removeEventListener("dragleave", onDragLeave);
  }
  window.removeEventListener("dragend", onDragLeave);
  draggable.unset();
});
</script>

<template>
  <div class="draggable" :class="{ dragging: movingItemIndex !== -1 }" ref="container">
    <div v-for="(item, index) in items" class="item" :key="item.id">
      <div class="handle" :data-index="index"><span class="icon-handle" /></div>
      <div class="content-wrapper">
        <div
          v-if="index === 0"
          class="drop-indicator top"
          :class="{ visible: dropIndicatorPosition === -1 }"
          :style="topDropIndicatorStyles()"
        />
        <div class="content" :class="{ moving: movingItemIndex === index }">
          <DynamicContentRasterizer :isRasterized="movingItemIndex !== -1">
            <slot name="item" v-bind="item"></slot>
          </DynamicContentRasterizer>
        </div>
        <div
          class="drop-indicator bottom"
          :class="{ visible: dropIndicatorPosition === index }"
          :style="dropIndicatorStyles(index)"
        />
      </div>
    </div>
    <div class="captured" v-if="movingItemAsPngData.length > 0" :style="capturedStyles()">
      <img :src="movingItemAsPngData" />
    </div>
  </div>
</template>

<style scoped>
.draggable {
  --content-left-margin: 18px;

  position: relative;
  display: flex;
  flex-direction: column;

  & .item {
    position: relative;
    width: 100%;
    transition: transform var(--animation-duration) var(--animation-easing);

    & .handle {
      position: absolute;
      top: 6px;
      left: 0;
      display: flex;
      color: var(--color-icon-primary);
      cursor: move;
      font-size: var(--font-size-lg);
      opacity: 0;
      transition: opacity var(--animation-duration) var(--animation-easing);
    }

    & .content-wrapper {
      width: calc(100% - var(--content-left-margin));
      margin-left: var(--content-left-margin);
    }

    & .content {
      width: 100%;
      transition: opacity var(--animation-duration) var(--animation-easing);

      &.moving {
        opacity: 0.4;
      }
    }

    & .drop-indicator {
      height: 0;
      border-radius: var(--radius-xs);
      background-color: var(--color-background-branding);
      opacity: 0;

      &.visible {
        height: 3px;
        opacity: 1;
      }

      &.bottom {
        margin: var(--spacing-16) 0;
      }
    }

    &:hover {
      & .handle {
        opacity: 1;
      }
    }
  }

  & .captured {
    position: absolute;
    z-index: 2;
    margin-left: var(--content-left-margin);
    box-shadow: 0 4px 17.7px 1px var(--color-shadow);
    transform-origin: top left;
  }

  &.dragging {
    * {
      touch-action: none;
      transition: none;
      user-select: none;
    }

    & .drop-indicator {
      transition:
        opacity var(--animation-duration) var(--animation-easing),
        height var(--animation-duration) var(--animation-easing),
        margin var(--animation-duration) var(--animation-easing);
    }
  }
}
</style>
