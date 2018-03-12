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
   ID_MIN_D         = 101,
   ID_MAX_D         = 102,
   ID_MIN_D_TEXT    = 103,
   ID_MAX_D_TEXT    = 104,
   ID_DISPLAY_COLOR = 105,
   ID_DISPLAY_DEPTH = 106,
   ID_DISPLAY_IR    = 107
};

wxDEFINE_EVENT(REFRESH_DISPLAY_EVENT, wxCommandEvent);

const size_t display_panel_width  = 576;
const size_t display_panel_height = 432;

// Declarations

class DisplayPanel;

class SettingsPanel : public wxPanel {
   DisplayPanel *picture_panel;

   wxPanel *m_parent;
   wxSlider *m_min_d, *m_max_d;
   wxTextCtrl *m_min_d_text, *m_max_d_text;

 public:
   SettingsPanel(wxPanel *parent, DisplayPanel *picture_panel);

   void on_change(wxCommandEvent &event);
};

class DisplayPanel : public wxPanel {
   wxStaticBitmap *m_picture = nullptr;

 public:
   DisplayPanel(wxPanel *parent, wxWindowID window_id, uint8_t *bitmap);

   void refresh_display(wxCommandEvent &event);

   uint8_t *bitmap;
};

class MainWindow : public wxFrame {
   Picture *picture;
   wxPanel *m_parent;

 public:
   explicit MainWindow(const wxString &title, uint8_t *color_bitmap, uint8_t *depth_bitmap, uint8_t *ir_bitmap);

   DisplayPanel *m_display_color, *m_display_depth, *m_display_ir;
   SettingsPanel *m_settings;
};

// Definitions

SettingsPanel::SettingsPanel(wxPanel *parent, DisplayPanel *picture_panel)
      : wxPanel(parent, -1, wxPoint(-1, -1), wxSize(-1, -1), wxBORDER_SUNKEN), picture_panel(picture_panel),
        m_parent(parent), m_min_d(new wxSlider(this, ID_MIN_D, 300, 0, 10000, wxPoint(60, 10), wxSize(980, 15))),
        m_max_d(new wxSlider(this, ID_MAX_D, 1000, 0, 10000, wxPoint(60, 40), wxSize(980, 15))),
        m_min_d_text(new wxTextCtrl(this, ID_MIN_D_TEXT, "300", wxPoint(10, 10), wxSize(40, 15))),
        m_max_d_text(new wxTextCtrl(this, ID_MAX_D_TEXT, "1000", wxPoint(10, 40), wxSize(40, 15))) {
   Connect(ID_MIN_D, wxEVT_SCROLL_CHANGED, wxCommandEventHandler(SettingsPanel::on_change));
   Connect(ID_MAX_D, wxEVT_SCROLL_CHANGED, wxCommandEventHandler(SettingsPanel::on_change));
}

void SettingsPanel::on_change(wxCommandEvent &event) {
   int64_t const min_depth = m_min_d->GetValue();
   int64_t const max_depth = m_max_d->GetValue();

   m_min_d_text->Clear();
   m_min_d_text->WriteText(std::to_string(m_min_d->GetValue()));
   m_max_d_text->Clear();
   m_max_d_text->WriteText(std::to_string(m_max_d->GetValue()));
   // TODO: Make wxTextCtrls non-editable or update values on edit.

   // picture_panel->update_picture(min_depth, max_depth);
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

MainWindow::MainWindow(const wxString &title, uint8_t *color_bitmap, uint8_t *depth_bitmap, uint8_t *ir_bitmap)
      : wxFrame(nullptr, wxID_ANY, title, wxDefaultPosition, wxSize(1900, 900)),
        picture(new Picture(nullptr, nullptr, nullptr)), m_parent(new wxPanel(this, wxID_ANY)),
        m_display_color(new DisplayPanel(m_parent, ID_DISPLAY_COLOR, color_bitmap)),
        m_display_depth(new DisplayPanel(m_parent, ID_DISPLAY_DEPTH, depth_bitmap)),
        m_display_ir(new DisplayPanel(m_parent, ID_DISPLAY_IR, ir_bitmap)),
        m_settings(new SettingsPanel(m_parent, m_display_color)) {

   auto vbox          = new wxBoxSizer(wxVERTICAL);
   auto displays_hbox = new wxBoxSizer(wxHORIZONTAL);
   displays_hbox->Add(m_display_color);
   displays_hbox->Add(m_display_depth);
   displays_hbox->Add(m_display_ir);
   vbox->Add(displays_hbox, 1, wxEXPAND | wxALL, 5);
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
   MainWindow *window;

 public:
   explicit MyKinectDevice(int device_number, MainWindow *window) : KinectDevice(device_number), window(window) {}

   void frame_handler(Picture const &picture) const override {
      Picture picture_copy(picture);

      if (picture_copy.color_frame) {
         auto frame_size = fit_to_size(picture_copy.color_frame->pixels->width,
               picture_copy.color_frame->pixels->height, display_panel_width, display_panel_height);
         size_t frame_width  = frame_size.first;
         size_t frame_height = frame_size.second;

         picture_copy.color_frame->resize(frame_width, frame_height);

         for (size_t i = 0; i < frame_height; ++i) {
            for (size_t j = 0; j < frame_width; ++j) {
               window->m_display_color->bitmap[3 * (i * display_panel_width + j)] =
                     (*picture_copy.color_frame->pixels)[i][j].red;
               window->m_display_color->bitmap[3 * (i * display_panel_width + j) + 1] =
                     (*picture_copy.color_frame->pixels)[i][j].green;
               window->m_display_color->bitmap[3 * (i * display_panel_width + j) + 2] =
                     (*picture_copy.color_frame->pixels)[i][j].blue;
            }
         }

         wxPostEvent(window->m_display_color, wxCommandEvent(REFRESH_DISPLAY_EVENT));
      }

      if (picture_copy.depth_frame) {
         auto frame_size = fit_to_size(picture_copy.depth_frame->pixels->width,
               picture_copy.depth_frame->pixels->height, display_panel_width, display_panel_height);
         size_t frame_width  = frame_size.first;
         size_t frame_height = frame_size.second;

         picture_copy.depth_frame->resize(frame_width, frame_height);

         auto int_pixels = new uint8_t[frame_width * frame_height];
         for (size_t i = 0; i < frame_width * frame_height; ++i) {
            int_pixels[i] = uint8_t(std::min(255.0 * picture_copy.depth_frame->pixels->data()[i] / 4500.0, 255.0));
         }

         cv::Mat current_image(
               cv::Size(static_cast<int>(frame_width), static_cast<int>(frame_height)), CV_8UC1, int_pixels);
         cv::Mat destination_image(cv::Size(static_cast<int>(frame_width), static_cast<int>(frame_height)), CV_8UC3);
         cv::applyColorMap(current_image, destination_image, cv::COLORMAP_HSV);

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

      if (picture_copy.ir_frame) {
         auto frame_size = fit_to_size(picture_copy.ir_frame->pixels->width, picture_copy.ir_frame->pixels->height,
               display_panel_width, display_panel_height);
         size_t frame_width  = frame_size.first;
         size_t frame_height = frame_size.second;

         picture_copy.ir_frame->resize(frame_width, frame_height);

         for (size_t i = 0; i < frame_height; ++i) {
            for (size_t j = 0; j < frame_width; ++j) {
               auto pixel_value = static_cast<uint8_t>(255.0 * (*picture_copy.ir_frame->pixels)[i][j] / 65535.0);
               window->m_display_ir->bitmap[3 * (i * display_panel_width + j)] = pixel_value;
               window->m_display_ir->bitmap[3 * (i * display_panel_width + j) + 1] = pixel_value;
               window->m_display_ir->bitmap[3 * (i * display_panel_width + j) + 2] = pixel_value;
            }
         }

         wxPostEvent(window->m_display_ir, wxCommandEvent(REFRESH_DISPLAY_EVENT));
      }
   }
};

// Main

class AppMain : public wxApp {
 public:
   bool OnInit() override;
};

IMPLEMENT_APP(AppMain)

bool AppMain::OnInit() {
   auto *color_bitmap = new uint8_t[display_panel_width * display_panel_height * 3];
   auto *depth_bitmap = new uint8_t[display_panel_width * display_panel_height * 3];
   auto *ir_bitmap    = new uint8_t[display_panel_width * display_panel_height * 3];
   for (size_t i = 0; i < display_panel_width * display_panel_height * 3; ++i) {
      color_bitmap[i] = 0;
      depth_bitmap[i] = 0;
      ir_bitmap[i]    = 0;
   }

   MainWindow *window = new MainWindow(wxT("Simple display"), color_bitmap, depth_bitmap, ir_bitmap);
   window->Show(true);

   auto kinect_device = new MyKinectDevice(0, window);
   kinect_device->start_streams(true, true, true);

   return true;
}
