<script setup lang="ts">
import LoadingBars from "@/components/LoadingBars.vue";
import ToastNotificationArea from "@/components/ToastNotificationArea.vue";
import { RouterView } from "vue-router";
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
  <ToastNotificationArea />
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
