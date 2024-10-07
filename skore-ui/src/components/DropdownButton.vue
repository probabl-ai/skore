<script lang="ts">
export interface DropdownProps {
  label?: string;
  icon?: string;
  isPrimary?: boolean;
  align?: "left" | "right";
  isInline?: boolean;
}
</script>

<script setup lang="ts">
import { autoUpdate, useFloating } from "@floating-ui/vue";
import { onBeforeUnmount, onMounted, ref } from "vue";

import SimpleButton from "@/components/SimpleButton.vue";

const { label, icon, isPrimary = false, align = "left", isInline } = defineProps<DropdownProps>();

const isOpen = ref(false);
const el = ref<HTMLDivElement>();
const reference = ref<HTMLElement>();
const floating = ref<HTMLDivElement>();
const { floatingStyles } = useFloating(reference, floating, {
  placement: align === "right" ? "bottom-end" : "bottom-start",
  strategy: "fixed",
  whileElementsMounted: autoUpdate,
});

function onClick(e: Event) {
  if (el.value && floating.value) {
    // is it a click outside or a click on an item ?
    if (!el.value.contains(e.target as Node) || floating.value.contains(e.target as Node)) {
      isOpen.value = false;
    }
  }
}

// Mouse move listener to close the dropdown when the mouse moves
function onMouseMove() {
  if (
    el.value &&
    !el.value.checkVisibility({
      opacityProperty: true,
      contentVisibilityAuto: true,
      visibilityProperty: true,
    })
  ) {
    isOpen.value = false;
  }
}

// Intersection observer to close the dropdown when it is not visible
const intersectionObserver = new IntersectionObserver(
  ([entry]) => {
    if (!entry.isIntersecting) {
      isOpen.value = false;
    }
  },
  {
    root: document.body,
    threshold: 0,
  }
);

onMounted(() => {
  document.addEventListener("click", onClick);
  document.addEventListener("mousemove", onMouseMove);
  if (el.value) {
    intersectionObserver.observe(el.value);
  }
});

onBeforeUnmount(() => {
  document.removeEventListener("click", onClick);
  document.removeEventListener("mousemove", onMouseMove);
  intersectionObserver.disconnect();
});
</script>

<template>
  <div class="dropdown" ref="el" :class="{ 'align-right': align === 'right' }">
    <SimpleButton
      :is-primary="isPrimary"
      :label="label"
      :icon="icon"
      :is-inline="isInline"
      @click="isOpen = !isOpen"
      ref="reference"
    />
    <Transition name="fade">
      <div class="items" v-if="isOpen" ref="floating" :style="floatingStyles">
        <slot></slot>
      </div>
    </Transition>
  </div>
</template>

<style scoped>
.dropdown {
  position: relative;
  display: inline-block;

  & .items {
    position: fixed;
    z-index: 9999;
    display: flex;
    overflow: visible;
    width: max-content;
    flex-direction: column;
    border: solid 1px var(--border-color-normal);
    border-radius: var(--border-radius);
    background-color: var(--background-color-normal);
    box-shadow: 4px 10px 20px var(--background-color-selected);
  }
}
</style>
