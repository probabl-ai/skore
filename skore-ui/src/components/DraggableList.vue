<script setup lang="ts">
import type { Interactable } from "@interactjs/types";
import { toPng } from "html-to-image";
import interact from "interactjs";
import { onMounted, onUnmounted, ref, useTemplateRef } from "vue";

interface Item {
  id: string;
  [key: string]: any;
}

const items = defineModel<Item[]>("items", { required: true });
const dropIndicatorPosition = ref<number | null>(null);
const movingItemIndex = ref(-1);
const movingItemAsPngData = ref("");
const movingItemHeight = ref(0);
const movingItemY = ref(0);
const container = useTemplateRef("container");
let interactable: Interactable;
let direction: "up" | "down" | "none" = "none";

function topDropIndicatorStyles() {
  if (dropIndicatorPosition.value === -1) {
    return {
      marginTop: `${movingItemHeight.value}px`,
      marginBottom: "var(--spacing-gap-normal)",
    };
  }
  return {};
}

function dropIndicatorStyles(index: number) {
  const y = dropIndicatorPosition.value === index ? movingItemHeight.value : 0;
  if (direction === "up") {
    return {
      marginTop: `calc(var(--spacing-gap-normal) + ${y}px)`,
    };
  } else if (direction === "down") {
    return {
      marginTop: `var(--spacing-gap-normal)`,
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

function makeModifier() {
  const containerBounds = container.value!.getBoundingClientRect();

  return interact.modifiers.restrict({
    restriction: (x, y, { element }) => {
      const content = element?.parentElement?.parentElement?.querySelector(".content");
      const { height } = content?.getBoundingClientRect() ?? { height: 0 };
      return {
        top: containerBounds?.top - height,
        right: containerBounds?.right,
        bottom: containerBounds?.bottom + height,
        left: containerBounds?.left,
      };
    },
  });
}

onMounted(() => {
  interactable = interact(".handle").draggable({
    autoScroll: true,
    startAxis: "y",
    lockAxis: "y",
    modifiers: [makeModifier()],
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

          console.log(
            `start drag done`,
            movingItemIndex.value,
            movingItemY.value,
            movingItemHeight.value
          );
        }
      },
      move(event) {
        // set direction
        direction = event.dy >= 0 ? "down" : "up";

        // move the rasterized copy
        movingItemY.value += event.dy;

        // set the drop indicator item index
        const itemBounds = Array.from(container.value!.querySelectorAll(".item")).map(
          (item, index) => {
            const { top, height } = item.getBoundingClientRect();
            const center = top + height / 2;
            return {
              index,
              distance: Math.abs(event.pageY - center),
            };
          }
        );
        const closestItemBelow = itemBounds.reduce((closest, item) => {
          if (item.distance < closest.distance) {
            return item;
          }
          return closest;
        }, itemBounds[0]);
        // if the first item is the closest we may need to move the drop indicator up
        if (closestItemBelow.index === 0) {
          // does the user want to move the item to the top?
          const bounds = container.value!.getBoundingClientRect();
          if (event.pageY < bounds.top) {
            dropIndicatorPosition.value = -1;
          } else {
            dropIndicatorPosition.value = 0;
          }
        } else {
          dropIndicatorPosition.value = closestItemBelow.index;
        }

        console.log(`move drag done`, dropIndicatorPosition.value, closestItemBelow.index);
      },
      end() {
        // change the model order
        if (items.value && movingItemIndex.value !== null && dropIndicatorPosition.value !== null) {
          // did user dropped the item in its previous position ?
          if (Math.abs(dropIndicatorPosition.value - movingItemIndex.value) > 1) {
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
});

onUnmounted(() => {
  interactable.unset();
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
          <slot name="item" v-bind="item"></slot>
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
@media (prefers-color-scheme: dark) {
  .draggable {
    --shadow-color: hsl(0deg 0% 54% / 25%);
  }
}

@media (prefers-color-scheme: light) {
  .draggable {
    --shadow-color: hsl(0deg 0% 46% / 25%);
  }
}

.draggable {
  --content-left-margin: 18px;

  position: relative;
  display: flex;
  flex-direction: column;

  & .item {
    position: relative;
    width: 100%;
    transition: transform var(--transition-duration) var(--transition-easing);

    & .handle {
      position: absolute;
      top: -4px;
      left: 0;
      display: flex;
      color: hsl(from var(--color-primary) h s calc(l * 1.5));
      cursor: move;
      font-size: calc(var(--text-size-normal) * 1.7);
      opacity: 0;
      transition: opacity var(--transition-duration) var(--transition-easing);
    }

    & .content-wrapper {
      width: calc(100% - var(--content-left-margin));
      margin-left: var(--content-left-margin);
    }

    & .content {
      width: 100%;
      transition: opacity var(--transition-duration) var(--transition-easing);

      &.moving {
        opacity: 0.4;
      }
    }

    & .drop-indicator {
      height: 0;
      border-radius: 3px;
      background-color: var(--color-primary);
      opacity: 0;
      transition:
        opacity var(--transition-duration) var(--transition-easing),
        height var(--transition-duration) var(--transition-easing),
        margin var(--transition-duration) var(--transition-easing);

      &.visible {
        height: 3px;
        opacity: 1;
      }

      &.bottom {
        margin: var(--spacing-gap-normal) 0;
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
    box-shadow: 0 4px 17.7px 1px var(--shadow-color);
    transform-origin: top left;
  }

  &.dragging {
    * {
      touch-action: none;
      transition: none;
      user-select: none;
    }
  }
}
</style>
