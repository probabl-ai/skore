<script setup lang="ts">
import { autoPlacement, autoUpdate, useFloating, type Placement } from "@floating-ui/vue";
import { computed, onBeforeUnmount, onMounted, ref } from "vue";

import SimpleButton, { type ButtonProps } from "@/components/SimpleButton.vue";

interface DropdownProps extends ButtonProps {
  allowedPlacements?: Placement[];
}

const props = withDefaults(defineProps<DropdownProps>(), {
  isPrimary: false,
  allowedPlacements: () => ["top-end", "bottom-end", "top-start", "bottom-start"],
});

const isOpen = ref(false);
const el = ref<HTMLDivElement>();
const reference = ref<HTMLElement>();
const floating = ref<HTMLDivElement>();
const { floatingStyles } = useFloating(reference, floating, {
  strategy: "fixed",
  middleware: [autoPlacement({ allowedPlacements: props.allowedPlacements as Placement[] })],
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

const transitionName = computed(() => {
  const isVisible = el.value?.checkVisibility({
    opacityProperty: true,
    contentVisibilityAuto: true,
    visibilityProperty: true,
  });
  return isVisible ? "fade" : "";
});

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
  <div class="dropdown" ref="el">
    <SimpleButton
      :is-primary="props.isPrimary"
      :label="props.label"
      :icon="props.icon"
      :is-inline="props.isInline"
      @click="isOpen = !isOpen"
      ref="reference"
    />
    <Transition :name="transitionName">
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
    border: solid var(--stroke-width-md) var(--color-stroke-background-primary);
    border-radius: var(--radius-xs);
    background-color: var(--color-background-primary);
    box-shadow: 4px 10px 20px var(--color-shadow);
  }
}
</style>
