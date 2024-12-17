<script setup lang="ts">
const isToggled = defineModel<boolean>("isToggled");
</script>

<template>
  <div class="slide-toggle" @click="isToggled = !isToggled" :class="{ toggled: isToggled }">
    <div class="thumb"></div>
  </div>
</template>

<style scoped>
.slide-toggle {
  --slide-toggle-height: 14px;

  position: relative;
  width: calc(var(--slide-toggle-height) * 2);
  height: var(--slide-toggle-height);
  border-radius: var(--slide-toggle-height);
  background-color: var(--color-icon-tertiary);
  transition: background-color var(--animation-duration) var(--animation-easing);

  & .thumb {
    --thumb-size: calc(var(--slide-toggle-height) - 2px);

    position: absolute;
    top: 1px;
    left: calc((var(--slide-toggle-height) * 2) - 1px - var(--thumb-size));
    width: var(--thumb-size);
    height: var(--thumb-size);
    border-radius: var(--thumb-size);
    background-color: white;
    transition: left var(--animation-duration) var(--animation-easing);
  }

  &.toggled {
    background-color: var(--color-icon-quartenary);

    & .thumb {
      left: 1px;
    }
  }
}
</style>
