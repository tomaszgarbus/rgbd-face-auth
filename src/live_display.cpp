#include <string>

#include <wx/bitmap.h>
#include <wx/event.h>
#include <wx/image.h>
#include <wx/panel.h>
#include <wx/slider.h>
#include <wx/stattext.h>
#include <wx/wx.h>
#include <wx/wxprec.h>

#include "libkinect.hpp"
#include "picture.hpp"

// Constants

enum {
   ID_MIN_D          = 101,
   ID_MAX_D          = 102,
   ID_MIN_D_TEXT     = 103,
   ID_MAX_D_TEXT     = 104,
   ID_DISPLAY_COLOR  = 105,
   ID_DISPLAY_DEPTH  = 106,
   ID_DISPLAY_IR     = 107,
   ID_DISPLAY_CUSTOM = 108
};

const size_t display_panel_width  = 512;
const size_t display_panel_height = 424;

wxDEFINE_EVENT(REFRESH_DISPLAY_EVENT, wxCommandEvent);

// Declarations

class DisplayPanel : public wxPanel {
   wxStaticBitmap *m_picture = nullptr;

 public:
   DisplayPanel(wxPanel *parent, wxWindowID window_id, uint8_t *bitmap);

   void refresh_display(wxCommandEvent &event);

   uint8_t *bitmap;
};

class SettingsPanel : public wxPanel {
 public:
   explicit SettingsPanel(wxPanel *parent);

   void on_min_slider_change(wxCommandEvent &event);
   void on_max_slider_change(wxCommandEvent &event);
   void on_min_text_change(wxCommandEvent &event);
   void on_max_text_change(wxCommandEvent &event);

   wxPanel *m_parent;
   wxSlider *m_min_d, *m_max_d;
   wxTextCtrl *m_min_d_text, *m_max_d_text;
};

class MainWindow : public wxFrame {
   wxPanel *m_parent;

 public:
   explicit MainWindow(const wxString &title, uint8_t *color_bitmap, uint8_t *depth_bitmap, uint8_t *ir_bitmap,
         uint8_t *custom_bitmap);

   DisplayPanel *m_display_color, *m_display_depth, *m_display_ir, *m_display_custom;
   SettingsPanel *m_settings;
   Picture *picture;
};

// Definitions

SettingsPanel::SettingsPanel(wxPanel *parent)
      : wxPanel(parent, -1, wxPoint(-1, -1), wxSize(-1, -1), wxBORDER_SUNKEN), m_parent(parent),
        m_min_d(new wxSlider(this, ID_MIN_D, 500, 0, 10000, wxPoint(80, 10), wxSize(980, 15))),
        m_max_d(new wxSlider(this, ID_MAX_D, 4500, 0, 10000, wxPoint(80, 40), wxSize(980, 15))),
        m_min_d_text(new wxTextCtrl(this, ID_MIN_D_TEXT, "500", wxPoint(10, 10), wxSize(60, 15))),
        m_max_d_text(new wxTextCtrl(this, ID_MAX_D_TEXT, "4500", wxPoint(10, 40), wxSize(60, 15))) {
   m_min_d->Bind(wxEVT_SCROLL_CHANGED, &SettingsPanel::on_min_slider_change, this);
   m_min_d->Bind(wxEVT_SCROLL_THUMBTRACK, &SettingsPanel::on_min_slider_change, this);
   m_max_d->Bind(wxEVT_SCROLL_CHANGED, &SettingsPanel::on_max_slider_change, this);
   m_max_d->Bind(wxEVT_SCROLL_THUMBTRACK, &SettingsPanel::on_max_slider_change, this);
   m_min_d_text->Bind(wxEVT_TEXT, &SettingsPanel::on_min_text_change, this);
   m_max_d_text->Bind(wxEVT_TEXT, &SettingsPanel::on_max_text_change, this);
}

void SettingsPanel::on_min_slider_change(wxCommandEvent &event) {
   if (m_max_d->GetValue() <= m_min_d->GetValue()) {
      m_max_d->SetValue(m_min_d->GetValue() + 1);
      m_max_d_text->Clear();
      m_max_d_text->WriteText(std::to_string(m_min_d->GetValue() + 1));
   }

   m_min_d_text->Clear();
   m_min_d_text->WriteText(std::to_string(m_min_d->GetValue()));
}

void SettingsPanel::on_max_slider_change(wxCommandEvent &event) {
   if (m_max_d->GetValue() <= m_min_d->GetValue()) {
      m_min_d->SetValue(m_max_d->GetValue() - 1);
      m_min_d_text->Clear();
      m_min_d_text->WriteText(std::to_string(m_max_d->GetValue() - 1));
   }

   m_max_d_text->Clear();
   m_max_d_text->WriteText(std::to_string(m_max_d->GetValue()));
}

void SettingsPanel::on_min_text_change(wxCommandEvent &event) {
   long min_d;
   if (!m_min_d_text->GetValue().ToLong(&min_d)) {
      return;
   }
   m_min_d->SetValue(static_cast<int>(min_d));

   if (m_max_d->GetValue() <= min_d) {
      m_max_d->SetValue(static_cast<int>(min_d) + 1);
      m_max_d_text->Clear();
      m_max_d_text->WriteText(std::to_string(min_d + 1));
   }
}

void SettingsPanel::on_max_text_change(wxCommandEvent &event) {
   long max_d;
   if (!m_max_d_text->GetValue().ToLong(&max_d)) {
      return;
   }
   m_max_d->SetValue(static_cast<int>(max_d));

   if (max_d <= m_min_d->GetValue()) {
      m_min_d->SetValue(static_cast<int>(max_d) - 1);
      m_min_d_text->Clear();
      m_min_d_text->WriteText(std::to_string(max_d - 1));
   }
}

DisplayPanel::DisplayPanel(wxPanel *parent, wxWindowID window_id, uint8_t *bitmap)
      : wxPanel(parent, window_id, wxPoint(0, 0), wxSize(display_panel_width, display_panel_height), wxBORDER_SUNKEN),
        bitmap(bitmap) {
   Bind(REFRESH_DISPLAY_EVENT, &DisplayPanel::refresh_display, this);
   wxPostEvent(this, wxCommandEvent(REFRESH_DISPLAY_EVENT));
}

void DisplayPanel::refresh_display(wxCommandEvent &event) {
   delete m_picture;
   m_picture = new wxStaticBitmap(this, wxID_ANY,
         wxBitmap(wxImage(display_panel_width, display_panel_height, bitmap, true)), wxDefaultPosition, wxDefaultSize);
}

MainWindow::MainWindow(
      const wxString &title, uint8_t *color_bitmap, uint8_t *depth_bitmap, uint8_t *ir_bitmap, uint8_t *custom_bitmap)
      : wxFrame(nullptr, wxID_ANY, title, wxDefaultPosition, wxSize(1500, 1000)),
        picture(new Picture(nullptr, nullptr, nullptr)), m_parent(new wxPanel(this, wxID_ANY)),
        m_display_color(new DisplayPanel(m_parent, ID_DISPLAY_COLOR, color_bitmap)),
        m_display_depth(new DisplayPanel(m_parent, ID_DISPLAY_DEPTH, depth_bitmap)),
        m_display_ir(new DisplayPanel(m_parent, ID_DISPLAY_IR, ir_bitmap)),
        m_display_custom(new DisplayPanel(m_parent, ID_DISPLAY_CUSTOM, custom_bitmap)),
        m_settings(new SettingsPanel(m_parent)) {
   auto hbox1 = new wxBoxSizer(wxHORIZONTAL), hbox2 = new wxBoxSizer(wxHORIZONTAL);
   hbox1->Add(m_display_color);
   hbox1->Add(m_display_depth);
   hbox2->Add(m_display_ir);
   hbox2->Add(m_display_custom);

   auto vbox = new wxBoxSizer(wxVERTICAL);
   vbox->Add(hbox1, 1, wxEXPAND | wxALL, 5);
   vbox->Add(hbox2, 1, wxEXPAND | wxALL, 5);
   vbox->Add(m_settings, 1, wxEXPAND | wxALL, 5);

   m_parent->SetSizer(vbox);
   Centre();
}

// Utils

std::pair<size_t, size_t> fit_to_size(size_t width, size_t height, size_t max_width, size_t max_height) {
   size_t new_height = max_width * height / width;
   if (new_height <= max_height) {
      return {max_width, new_height};
   } else {
      return {max_height * width / height, max_height};
   }
}

// Kinect handling

class MyKinectDevice : public KinectDevice {
 public:
   explicit MyKinectDevice(int device_number) : KinectDevice(device_number) {}

   void frame_handler(Picture const &picture) const override;

   MainWindow *window = nullptr;
};

void MyKinectDevice::frame_handler(Picture const &picture) const {
   if (window == nullptr) {
      return;
   }

   if (picture.color_frame) {  // sorry
      delete window->picture->color_frame;
      window->picture->color_frame = new Picture::ColorFrame(*picture.color_frame);

      auto frame_size = fit_to_size(window->picture->color_frame->pixels->width,
            window->picture->color_frame->pixels->height, display_panel_width, display_panel_height);
      size_t frame_width  = frame_size.first;
      size_t frame_height = frame_size.second;

      if (frame_width != window->picture->color_frame->pixels->width
            || frame_height != window->picture->color_frame->pixels->height) {
         window->picture->color_frame->resize(frame_width, frame_height);
      }

      for (size_t i = 0; i < frame_height; ++i) {
         for (size_t j = 0; j < frame_width; ++j) {
            window->m_display_color->bitmap[3 * (i * display_panel_width + j)] =
                  (*window->picture->color_frame->pixels)[i][j].red;
            window->m_display_color->bitmap[3 * (i * display_panel_width + j) + 1] =
                  (*window->picture->color_frame->pixels)[i][j].green;
            window->m_display_color->bitmap[3 * (i * display_panel_width + j) + 2] =
                  (*window->picture->color_frame->pixels)[i][j].blue;
         }
      }

      wxPostEvent(window->m_display_color, wxCommandEvent(REFRESH_DISPLAY_EVENT));
   }

   if (picture.depth_frame) {
      delete window->picture->depth_frame;
      window->picture->depth_frame = new Picture::DepthOrIrFrame(*picture.depth_frame);

      auto frame_size = fit_to_size(window->picture->depth_frame->pixels->width,
            window->picture->depth_frame->pixels->height, display_panel_width, display_panel_height);
      size_t frame_width  = frame_size.first;
      size_t frame_height = frame_size.second;

      if (frame_width != window->picture->depth_frame->pixels->width
            || frame_height != window->picture->depth_frame->pixels->height) {
         window->picture->depth_frame->resize(frame_width, frame_height);
      }

      float min_depth = window->m_settings->m_min_d->GetValue(), max_depth = window->m_settings->m_max_d->GetValue();
      if (max_depth - min_depth < 1.0) {
         max_depth = min_depth + 1.0f;
      }

      auto int_pixels = new uint8_t[frame_width * frame_height];
      for (size_t i = 0; i < frame_width * frame_height; ++i) {
         int_pixels[i] = uint8_t(std::max(0.0,
               std::min(255.0 * (window->picture->depth_frame->pixels->data()[i] - min_depth) / (max_depth - min_depth),
                                                255.0)));
      }

      cv::Mat current_image(
            cv::Size(static_cast<int>(frame_width), static_cast<int>(frame_height)), CV_8UC1, int_pixels);
      cv::Mat destination_image(cv::Size(static_cast<int>(frame_width), static_cast<int>(frame_height)), CV_8UC3);
      cv::applyColorMap(current_image, destination_image, cv::COLORMAP_RAINBOW);

      for (size_t i = 0; i < frame_height; ++i) {
         for (size_t j = 0; j < frame_width; ++j) {
            auto pixel = destination_image.at<cv::Vec3b>(static_cast<int>(i), static_cast<int>(j));
            window->m_display_depth->bitmap[3 * (i * display_panel_width + j)]     = pixel[2];
            window->m_display_depth->bitmap[3 * (i * display_panel_width + j) + 1] = pixel[1];
            window->m_display_depth->bitmap[3 * (i * display_panel_width + j) + 2] = pixel[0];
         }
      }

      wxPostEvent(window->m_display_depth, wxCommandEvent(REFRESH_DISPLAY_EVENT));
   }

   if (picture.ir_frame) {
      delete window->picture->ir_frame;
      window->picture->ir_frame = new Picture::DepthOrIrFrame(*picture.ir_frame);

      auto frame_size = fit_to_size(window->picture->ir_frame->pixels->width, window->picture->ir_frame->pixels->height,
            display_panel_width, display_panel_height);
      size_t frame_width  = frame_size.first;
      size_t frame_height = frame_size.second;

      if (frame_width != window->picture->ir_frame->pixels->width
            || frame_height != window->picture->ir_frame->pixels->height) {
         window->picture->ir_frame->resize(frame_width, frame_height);
      }

      float max_value;
      if (which_kinect == 1) {
         max_value = 1024.0;
      } else {
         max_value = 65535.0;
      }

      for (size_t i = 0; i < frame_height; ++i) {
         for (size_t j = 0; j < frame_width; ++j) {
            auto pixel_value = static_cast<uint8_t>(255.0 * (*window->picture->ir_frame->pixels)[i][j] / max_value);
            window->m_display_ir->bitmap[3 * (i * display_panel_width + j)]     = pixel_value;
            window->m_display_ir->bitmap[3 * (i * display_panel_width + j) + 1] = pixel_value;
            window->m_display_ir->bitmap[3 * (i * display_panel_width + j) + 2] = pixel_value;
         }
      }

      wxPostEvent(window->m_display_ir, wxCommandEvent(REFRESH_DISPLAY_EVENT));
   }

   if ((picture.depth_frame || picture.ir_frame) && window->picture->depth_frame && window->picture->ir_frame
         && window->picture->depth_frame->pixels->width == window->picture->ir_frame->pixels->width
         && window->picture->depth_frame->pixels->height == window->picture->ir_frame->pixels->height) {
      auto frame_width  = window->picture->depth_frame->pixels->width,
           frame_height = window->picture->depth_frame->pixels->height;

      auto values      = new double[frame_width * frame_height];
      double max_value = 0.0;

      for (size_t i = 0; i < frame_height; ++i) {
         for (size_t j = 0; j < frame_width; ++j) {
            double distance             = (*window->picture->depth_frame->pixels)[i][j];
            values[i * frame_width + j] = distance * distance * (*window->picture->ir_frame->pixels)[i][j];
            max_value                   = std::max(max_value, values[i * frame_width + j]);
         }
      }

      // std::cerr << max_value << '\n';
      max_value = 4e10;

      for (size_t i = 0; i < frame_height; ++i) {
         for (size_t j = 0; j < frame_width; ++j) {
            auto pixel_value = static_cast<uint8_t>(std::min(255.0, 255.0 * values[i * frame_width + j] / max_value));
            float min_red    = 255.0f * static_cast<float>(window->m_settings->m_min_d->GetValue()) / 10000.0f;
            float max_red    = 255.0f * static_cast<float>(window->m_settings->m_max_d->GetValue()) / 10000.0f;
            if (pixel_value >= min_red && pixel_value <= max_red) {
               window->m_display_custom->bitmap[3 * (i * display_panel_width + j)]     = 255;
               window->m_display_custom->bitmap[3 * (i * display_panel_width + j) + 1] = 0;
               window->m_display_custom->bitmap[3 * (i * display_panel_width + j) + 2] = 0;
            } else {
               window->m_display_custom->bitmap[3 * (i * display_panel_width + j)]     = pixel_value;
               window->m_display_custom->bitmap[3 * (i * display_panel_width + j) + 1] = pixel_value;
               window->m_display_custom->bitmap[3 * (i * display_panel_width + j) + 2] = pixel_value;
            }
         }
      }

      delete values;

      wxPostEvent(window->m_display_custom, wxCommandEvent(REFRESH_DISPLAY_EVENT));
   }
}

// Main

class AppMain : public wxApp {
 public:
   bool OnInit() override;
   MyKinectDevice *kinect_device = nullptr;
};

bool AppMain::OnInit() {
   auto *color_bitmap  = new uint8_t[display_panel_width * display_panel_height * 3];
   auto *depth_bitmap  = new uint8_t[display_panel_width * display_panel_height * 3];
   auto *ir_bitmap     = new uint8_t[display_panel_width * display_panel_height * 3];
   auto *custom_bitmap = new uint8_t[display_panel_width * display_panel_height * 3];
   for (size_t i = 0; i < display_panel_width * display_panel_height * 3; ++i) {
      color_bitmap[i]  = 0;
      depth_bitmap[i]  = 0;
      ir_bitmap[i]     = 0;
      custom_bitmap[i] = 0;
   }

   MainWindow *window =
         new MainWindow(wxT("Live Kinect display"), color_bitmap, depth_bitmap, ir_bitmap, custom_bitmap);
   window->Show(true);

   kinect_device->window = window;

   return true;
}

int main(int argc, char **argv) {
   auto kinect_device = new MyKinectDevice(0);
   bool use_color, use_depth, use_ir;
   if (kinect_device->which_kinect == 1) {
      use_color = false;
      use_depth = true;
      use_ir    = true;
   } else {
      use_color = true;
      use_depth = true;
      use_ir    = true;
   }
   kinect_device->start_streams(use_color, use_depth, use_ir);

   auto app           = new AppMain();
   app->kinect_device = kinect_device;
   wxApp::SetInstance(app);
   return wxEntry(argc, argv);
}