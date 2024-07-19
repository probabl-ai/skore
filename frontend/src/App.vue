<script setup lang="ts">
import { RouterView } from "vue-router";
import LoadingBars from "./components/LoadingBars.vue";
</script>

<template>
  <RouterView v-slot="{ Component }">
    <template v-if="Component">
      <Transition name="fade">
        <Suspense>
          <component :is="Component" />
          <template #fallback>
            <div class="loader">
              <LoadingBars />
            </div>
          </template>
        </Suspense>
      </Transition>
    </template>
  </RouterView>
</template>

<style scoped>
.loader {
  display: flex;
  width: 100dvw;
  height: 100dvh;
  align-items: center;
  justify-content: center;
}
</style>
