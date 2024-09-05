<script setup lang="ts">
import SimpleButton from "@/components/SimpleButton.vue";
import { onBeforeUnmount, onMounted, ref } from "vue";

interface DropdownProps {
  label?: string;
  icon?: string;
  isPrimary?: boolean;
  align?: "left" | "right";
}

const props = withDefaults(defineProps<DropdownProps>(), { isPrimary: false, align: "left" });

const isOpen = ref(false);
const el = ref<HTMLDivElement>();

function closeDropdown(e: Event) {
  if (el.value && !el.value.contains(e.target as Node)) {
    isOpen.value = false;
  }
}

onMounted(() => {
  document.addEventListener("click", closeDropdown);
});

onBeforeUnmount(() => {
  document.removeEventListener("click", closeDropdown);
});
</script>

<template>
  <div class="dropdown" ref="el" :class="{ 'align-right': props.align === 'right' }">
    <SimpleButton
      :is-primary="props.isPrimary"
      :label="props.label"
      :icon="props.icon"
      @click="isOpen = !isOpen"
    />
    <Transition name="fade">
      <div class="items" v-if="isOpen">
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
    position: absolute;
    z-index: 1000;
    top: 100%;
    left: 0;
    display: flex;
    overflow: visible;
    flex-direction: column;
    padding: var(--spacing-padding-small);
    border: solid 1px var(--border-color-normal);
    border-radius: var(--border-radius);
    background-color: var(--background-color-normal);
    box-shadow: 4px 10px 20px var(--background-color-selected);
    gap: var(--spacing-padding-small);
  }

  &.align-right {
    & .items {
      right: 0;
      left: unset;
    }
  }
}
</style>
