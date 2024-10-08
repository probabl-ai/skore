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
const dropIndicatorPosition = ref(-1);
const movingItemIndex = ref(-1);
const movingItemAsPngData = ref("");
const movingItemHeight = ref(0);
const movingItemY = ref(0);
const container = useTemplateRef("container");
let interactable: Interactable;
let direction: "up" | "down" | "none" = "none";

function makeModifier() {
  const containerBounds = container.value!.getBoundingClientRect();

  return interact.modifiers.restrict({
    restriction: (x, y, { element }) => {
      const content = element?.parentElement?.querySelector(".content");
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

function dropIndicatorStyles(index: number) {
  const y = dropIndicatorPosition.value === index ? movingItemHeight.value : 0;
  if (direction === "up") {
    return {
      marginTop: `calc(var(--spacing-gap-normal) + ${y}px)`,
    };
  } else if (direction === "down") {
    return {
      marginTop: `calc(var(--spacing-gap-normal)`,
      marginBottom: `${y}px`,
    };
  }
}
onMounted(() => {
  interactable = interact(".handle").draggable({
    startAxis: "y",
    lockAxis: "y",
    modifiers: [makeModifier()],
    listeners: {
      async start(event) {
        // make a rasterized copy of the moving element
        const content = event.target.parentElement.querySelector(".content");
        if (content && container.value) {
          movingItemAsPngData.value = await toPng(content);
          movingItemIndex.value = parseInt(event.target.dataset.index);
          const parentBounds = container.value.getBoundingClientRect();
          const bounds = content.getBoundingClientRect();
          movingItemY.value = bounds.top - parentBounds.top;
          movingItemHeight.value = bounds.height;
        }
      },
      move(event) {
        direction = event.dy >= 0 ? "down" : "up";
        // move the rasterized copy
        movingItemY.value += event.dy;
        // compute the drop indicator position
        const parentBounds = container.value!.getBoundingClientRect();
        const element = container.value!.querySelectorAll(".item");
        dropIndicatorPosition.value = -1;
        for (let i = 0; i < element.length; i++) {
          const item = element[i];
          const itemBounds = item.getBoundingClientRect();
          const itemTop = itemBounds.top - parentBounds.top;
          const itemBottom = itemBounds.bottom - parentBounds.top;
          if (movingItemY.value >= itemTop && movingItemY.value <= itemBottom) {
            dropIndicatorPosition.value = i;
            break;
          }
        }
      },
      end() {
        // change the model order
        if (items.value) {
          // move the item to its new position
          const newItems = items.value.filter((_, index) => index !== movingItemIndex.value);
          newItems.splice(dropIndicatorPosition.value, 0, items.value[movingItemIndex.value]);
          items.value = newItems;
        }
        movingItemIndex.value = -1;
        dropIndicatorPosition.value = -1;
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
    <ul class="debug">
      <li>movingItemIndex: {{ movingItemIndex }}</li>
      <li>dropIndicatorPosition: {{ dropIndicatorPosition }}</li>
      <li>items: {{ items?.map((item) => item.id).join(", ") }}</li>
    </ul>

    <div v-for="(item, index) in items" class="item" :key="item.id">
      <div class="handle" :data-index="index"><span class="icon-handle" /></div>
      <div class="content" :class="{ moving: movingItemIndex === index }">
        <slot name="item" v-bind="item"></slot>
      </div>
      <div
        class="drop-indicator"
        :class="{ visible: dropIndicatorPosition === index }"
        :style="dropIndicatorStyles(index)"
      />
    </div>
    <div
      class="captured"
      v-if="movingItemAsPngData.length > 0"
      :style="{ transform: `translateY(${movingItemY}px) rotate(-4deg)` }"
    >
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
  position: relative;
  display: flex;
  flex-direction: column;
  gap: var(--spacing-gap-normal);

  & .item {
    position: relative;
    width: 100%;
    transition: transform var(--transition-duration) var(--transition-easing);

    & .handle {
      position: absolute;
      top: 0;
      left: -20px;
      cursor: move;
    }

    & .content {
      transform-origin: top left;
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
        margin-top var(--transition-duration) var(--transition-easing);

      &.visible {
        height: 3px;
        opacity: 1;
      }
    }
  }

  & .captured {
    position: absolute;
    z-index: 2;
    box-shadow: 0 4px 17.7px 1px var(--shadow-color);
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
